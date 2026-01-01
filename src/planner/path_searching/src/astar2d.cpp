#include <path_searching/astar2d.h>
#include <plan_env/risk_map2d.h>
#include <sstream>

using namespace std;
using namespace Eigen;

namespace apexnav_planner {
Astar2D::~Astar2D()
{
  for (int i = 0; i < allocate_num_; i++) delete path_node_pool_[i];
}

void Astar2D::init(ros::NodeHandle& nh, const SDFMap2D::Ptr& sdf_map)
{
  nh.param("astar/resolution_astar", resolution_, -1.0);
  nh.param("astar/lambda_heu", lambda_heu_, -1.0);
  nh.param("astar/risk_weight", lambda_risk_, 0.0);
  allocate_num_ = 1000000;

  this->sdf_map_ = sdf_map;

  /* ---------- map params ---------- */
  this->inv_resolution_ = 1.0 / resolution_;
  sdf_map_->getRegion(origin_, map_size_2d_);
  cout << "origin_: " << origin_.transpose() << endl;
  cout << "map size: " << map_size_2d_.transpose() << endl;

  path_node_pool_.resize(allocate_num_);
  for (int i = 0; i < allocate_num_; i++) path_node_pool_[i] = new Node2D;
  use_node_num_ = 0;
  iter_num_ = 0;
  early_terminate_cost_ = 0.0;
}

void Astar2D::reset()
{
  open_set_map_.clear();
  close_set_map_.clear();
  path_nodes_.clear();

  std::priority_queue<Node2DPtr, std::vector<Node2DPtr>, NodeComparator2D> empty_queue;
  open_set_.swap(empty_queue);
  for (int i = 0; i < use_node_num_; i++) {
    path_node_pool_[i]->parent = nullptr;
  }
  use_node_num_ = 0;
  iter_num_ = 0;
}

void Astar2D::setResolution(const double& res)
{
  resolution_ = res;
  this->inv_resolution_ = 1.0 / resolution_;
}

// JPS (Jump Point Search) implementation - maintains same interface as A*
int Astar2D::astarSearch(const Eigen::Vector2d& start_pt, const Eigen::Vector2d& end_pt,
    double success_dist, double max_time, int safety_mode)
{
  Node2DPtr cur_node = path_node_pool_[0];
  cur_node->parent = nullptr;
  cur_node->position = start_pt;
  posToIndex(start_pt, cur_node->index);
  cur_node->g_score = 0.0;
  cur_node->f_score = lambda_heu_ * getDiagHeu(cur_node->position, end_pt);

  Eigen::Vector2i end_index;
  posToIndex(end_pt, end_index);

  open_set_.push(cur_node);
  open_set_map_.insert(make_pair(cur_node->index, cur_node));
  use_node_num_ += 1;

  const auto t1 = ros::Time::now();

  /* ---------- JPS search loop ---------- */
  while (!open_set_.empty()) {
    // Get node with lowest f_score
    cur_node = open_set_.top();
    open_set_.pop();
    
    // Skip if already in close set (can happen due to duplicate entries in priority queue)
    if (close_set_map_.find(cur_node->index) != close_set_map_.end())
      continue;
    
    bool reach_end =
        abs(cur_node->index(0) - end_index(0)) <= 1 && abs(cur_node->index(1) - end_index(1)) <= 1;
    if ((cur_node->position - end_pt).norm() < success_dist)
      reach_end = true;
    if (reach_end) {
      backtrack(cur_node, end_pt);
      return REACH_END;
    }

    // Early termination if time up
    if ((ros::Time::now() - t1).toSec() > max_time) {
      early_terminate_cost_ = cur_node->g_score + getDiagHeu(cur_node->position, end_pt);
      return NO_PATH;
    }

    // Mark as closed
    open_set_map_.erase(cur_node->index);
    close_set_map_.insert(make_pair(cur_node->index, 1));
    iter_num_ += 1;

    Eigen::Vector2i cur_idx = cur_node->index;
    Eigen::Vector2d cur_pos = cur_node->position;

    // JPS: Use 8-direction grid-based jumping (JPS core algorithm)
    // Note: JPS algorithm is designed for 8-direction grid movement (standard implementation)
    // The 12-direction capability from generateSteps() is preserved in:
    //   1. Path cost calculation (uses actual Euclidean distance, not grid distance)
    //   2. Path output (continuous positions, not just grid centers)
    //   3. Safety checks (raycast uses continuous space)
    // The 8-direction grid search is just for efficiently finding jump points
    // 8 directions for grid: N, NE, E, SE, S, SW, W, NW
    const int dirs[8][2] = {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
    
    // Determine parent direction for JPS pruning (in grid space)
    Eigen::Vector2i parent_dir(0, 0);
    if (cur_node->parent != nullptr) {
      parent_dir = cur_idx - cur_node->parent->index;
      // Normalize to unit grid direction
      if (parent_dir(0) != 0) parent_dir(0) = parent_dir(0) / abs(parent_dir(0));
      if (parent_dir(1) != 0) parent_dir(1) = parent_dir(1) / abs(parent_dir(1));
    }

    // For each of 8 grid directions, find jump points (JPS core)
    for (int d = 0; d < 8; ++d) {
      Eigen::Vector2i grid_dir(dirs[d][0], dirs[d][1]);
      
      // JPS pruning: skip symmetric directions
      // For start node (no parent), check all directions
      // For other nodes, only check "natural" directions based on parent
      if (cur_node->parent != nullptr) {
        bool is_natural = false;
        
        // Same direction as parent is always natural
        if (grid_dir(0) == parent_dir(0) && grid_dir(1) == parent_dir(1)) {
          is_natural = true;
        } 
        // Diagonal parent: allow same diagonal, and both cardinal components
        else if (parent_dir(0) != 0 && parent_dir(1) != 0) {
          if ((grid_dir(0) == parent_dir(0) && grid_dir(1) == 0) || 
              (grid_dir(0) == 0 && grid_dir(1) == parent_dir(1))) {
            is_natural = true;
          }
        } 
        // Cardinal parent: allow perpendicular diagonals
        else {
          // Parent is cardinal (e.g., North), allow diagonals that include that direction
          if (parent_dir(0) == 0) {
            // Parent is vertical (N or S), allow diagonals with same vertical component
            if (grid_dir(1) == parent_dir(1) && grid_dir(0) != 0) {
              is_natural = true;
            }
          } else {
            // Parent is horizontal (E or W), allow diagonals with same horizontal component
            if (grid_dir(0) == parent_dir(0) && grid_dir(1) != 0) {
              is_natural = true;
            }
          }
        }
        
        if (!is_natural) continue;
      }
      // If no parent (start node), check all directions (is_natural remains true by default)

      // Jump in this grid direction
      Eigen::Vector2i jump_point = jump(cur_idx, grid_dir, end_index, safety_mode);
      
      if (jump_point(0) < 0) continue;  // No valid jump point

      // Convert jump point index to position
      Eigen::Vector2d jump_pos;
      jump_pos(0) = origin_(0) + (jump_point(0) + 0.5) * resolution_;
      jump_pos(1) = origin_(1) + (jump_point(1) + 0.5) * resolution_;

      // Check if jump point is safe (with raycast check)
      if ((jump_pos - start_pt).norm() > 0.25) {
        if (!checkPointSafety(jump_pos, safety_mode))
          continue;

        // Raycast safety check
        bool safe = true;
        Vector2d dir_vec = jump_pos - cur_pos;
        double len = dir_vec.norm();
        if (len > 1e-3) {
          dir_vec.normalize();
          for (double l = 0.025; l < len; l += 0.025) {
            Vector2d ckpt = cur_pos + l * dir_vec;
            if (!checkPointSafety(ckpt, safety_mode)) {
              safe = false;
              break;
            }
          }
        }
        if (!safe) continue;
      }

      // Check if already in close set
      if (close_set_map_.find(jump_point) != close_set_map_.end())
        continue;

      // Calculate cost with risk consideration
      double step_length = (jump_pos - cur_pos).norm();
      double risk = 0.0;
      if (sdf_map_->risk_map_) {
        risk = sdf_map_->risk_map_->get(jump_pos);
        if (std::isnan(risk) || risk < 0.0) risk = 0.0;
      }
      double step_cost = step_length * (1.0 + lambda_risk_ * risk);
      double tmp_g_score = step_cost + cur_node->g_score;

      Node2DPtr neighbor;
      auto node_iter = open_set_map_.find(jump_point);
      if (node_iter == open_set_map_.end()) {
        neighbor = path_node_pool_[use_node_num_];
        use_node_num_ += 1;
        if (use_node_num_ == allocate_num_) {
          cout << "run out of node pool." << endl;
          return NO_PATH;
        }
        neighbor->index = jump_point;
        neighbor->position = jump_pos;
      }
      else if (tmp_g_score < node_iter->second->g_score) {
        neighbor = node_iter->second;
      }
      else
        continue;

      neighbor->parent = cur_node;
      neighbor->g_score = tmp_g_score;
      neighbor->f_score = tmp_g_score + lambda_heu_ * getDiagHeu(jump_pos, end_pt);
      open_set_.push(neighbor);
      open_set_map_[jump_point] = neighbor;
    }
  }
  return NO_PATH;
}

std::vector<Eigen::Vector2d> Astar2D::generateSteps(Eigen::Vector2d pos)
{
  vector<Eigen::Vector2d> steps;

  // Normal Astar Step
  // for (double dx = -resolution_; dx <= resolution_ + 1e-3; dx += resolution_)
  //   for (double dy = -resolution_; dy <= resolution_ + 1e-3; dy += resolution_) {
  //     Eigen::Vector2d step;
  //     step << dx, dy;
  //     if (step.norm() < 1e-3)
  //       continue;
  //     steps.push_back(step);
  //   }

  // Habitat-like stepping (12 directions)
  const double step_length = 0.25;
  const double angle_increment = M_PI / 6;

  for (int i = 0; i < 12; ++i) {
    double angle = i * angle_increment;
    Eigen::Vector2d step(step_length * cos(angle), step_length * sin(angle));
    steps.push_back(step);
  }
  return steps;
}

double Astar2D::getDiagHeu(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
{
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double tie_breaker = 1.0 + 1e-6 * (dx + dy);
  // Diagonal distance heuristic for 2D
  return tie_breaker * (sqrt(2.0) * min(dx, dy) + abs(dx - dy));
}

double Astar2D::getManhHeu(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
{
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double tie_breaker = 1.0 + 1e-6 * (dx + dy);
  // Manhattan distance heuristic for 2D
  return tie_breaker * (dx + dy);
}

double Astar2D::getEuclHeu(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
{
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double tie_breaker = 1.0 + 1e-6 * (dx + dy);
  // Euclidean distance heuristic for 2D
  return tie_breaker * (x2 - x1).norm();
}

void Astar2D::backtrack(const Node2DPtr& end_node, const Eigen::Vector2d& end)
{
  path_nodes_.push_back(end);
  path_nodes_.push_back(end_node->position);
  Node2DPtr cur_node = end_node;
  while (cur_node->parent != nullptr) {
    cur_node = cur_node->parent;
    path_nodes_.push_back(cur_node->position);
  }
  reverse(path_nodes_.begin(), path_nodes_.end());
}

std::vector<Eigen::Vector2d> Astar2D::getVisited()
{
  vector<Eigen::Vector2d> visited;
  for (int i = 0; i < use_node_num_; ++i) visited.push_back(path_node_pool_[i]->position);
  return visited;
}

void Astar2D::posToIndex(const Eigen::Vector2d& pt, Eigen::Vector2i& idx)
{
  idx = ((pt - origin_) * inv_resolution_).array().floor().cast<int>();
}

std::vector<Eigen::Vector2d> Astar2D::getPath()
{
  return path_nodes_;
}

double Astar2D::pathLength(const vector<Eigen::Vector2d>& path)
{
  double length = 0.0;
  if (path.size() < 2)
    return length;
  for (int i = 0; i < (int)path.size() - 1; ++i) length += (path[i + 1] - path[i]).norm();
  return length;
}

bool Astar2D::checkPointSafety(const Eigen::Vector2d& pos, int safety_mode)
{
  // Outside map bounds is always unsafe
  if (!sdf_map_->isInMap(pos))
    return false;

  // EXTREME: allow any position inside the map (bypass occupancy checks)
  if (safety_mode == SAFETY_MODE::EXTREME)
    return true;

  // Occupancy checks
  const auto occ = sdf_map_->getOccupancy(pos);
  // If inflated occupancy marks collision, or cell is definitely occupied -> unsafe
  if (sdf_map_->getInflateOccupancy(pos) == 1 || occ == SDFMap2D::OCCUPIED)
    return false;

  // In NORMAL mode, treat unknown as unsafe. In OPTIMISTIC, unknown is allowed.
  if (occ == SDFMap2D::UNKNOWN && safety_mode == SAFETY_MODE::NORMAL)
    return false;

  return true;
}

// JPS: Check if a grid cell is walkable
bool Astar2D::isWalkable(const Eigen::Vector2i& idx, int safety_mode)
{
  Eigen::Vector2d pos;
  pos(0) = origin_(0) + (idx(0) + 0.5) * resolution_;
  pos(1) = origin_(1) + (idx(1) + 0.5) * resolution_;
  return checkPointSafety(pos, safety_mode);
}

// JPS: Check if current node has forced neighbors (defines jump points)
// Forced neighbors: when moving in direction dir, if there's an obstacle in a perpendicular
// direction that forces us to consider a neighbor in that direction
bool Astar2D::hasForcedNeighbor(const Eigen::Vector2i& cur, const Eigen::Vector2i& dir, int safety_mode)
{
  if (dir(0) == 0 || dir(1) == 0) {
    // Cardinal direction (N, S, E, W)
    // Forced neighbor occurs when:
    // - There's an obstacle in the perpendicular direction ahead
    // - But the neighbor in that perpendicular direction is walkable
    
    Eigen::Vector2i perp1(-dir(1), dir(0));  // Left perpendicular
    Eigen::Vector2i perp2(dir(1), -dir(0));  // Right perpendicular
    
    // Check left side: obstacle ahead in left direction, but left neighbor is free
    Eigen::Vector2i left_ahead = cur + dir + perp1;
    Eigen::Vector2i left_neighbor = cur + perp1;
    if (!isWalkable(left_ahead, safety_mode) && isWalkable(left_neighbor, safety_mode)) {
      return true;
    }
    
    // Check right side: obstacle ahead in right direction, but right neighbor is free
    Eigen::Vector2i right_ahead = cur + dir + perp2;
    Eigen::Vector2i right_neighbor = cur + perp2;
    if (!isWalkable(right_ahead, safety_mode) && isWalkable(right_neighbor, safety_mode)) {
      return true;
    }
  } else {
    // Diagonal direction (NE, SE, SW, NW)
    // Forced neighbor occurs when:
    // - One of the cardinal components is blocked
    // - But the diagonal direction and the forced neighbor position are walkable
    
    Eigen::Vector2i card1(dir(0), 0);  // Horizontal component
    Eigen::Vector2i card2(0, dir(1));  // Vertical component
    
    // Check if diagonal is walkable (required for forced neighbor)
    if (!isWalkable(cur + dir, safety_mode))
      return false;
    
    // Case 1: Horizontal cardinal blocked, but forced neighbor (horizontal + diagonal) is free
    if (!isWalkable(cur + card1, safety_mode)) {
      Eigen::Vector2i forced = cur + card1 + dir;
      if (isWalkable(forced, safety_mode))
        return true;
    }
    
    // Case 2: Vertical cardinal blocked, but forced neighbor (vertical + diagonal) is free
    if (!isWalkable(cur + card2, safety_mode)) {
      Eigen::Vector2i forced = cur + card2 + dir;
      if (isWalkable(forced, safety_mode))
        return true;
    }
  }
  
  return false;
}

// JPS: Jump in a direction until we hit a jump point or obstacle (iterative version)
Eigen::Vector2i Astar2D::jump(const Eigen::Vector2i& cur, const Eigen::Vector2i& dir,
    const Eigen::Vector2i& goal, int safety_mode)
{
  Eigen::Vector2i current = cur;
  const int max_jump_distance = 1000;  // Safety limit
  int jump_count = 0;
  
  while (jump_count < max_jump_distance) {
    Eigen::Vector2i next = current + dir;
    jump_count++;
    
    // Check bounds
    Eigen::Vector2d next_pos;
    next_pos(0) = origin_(0) + (next(0) + 0.5) * resolution_;
    next_pos(1) = origin_(1) + (next(1) + 0.5) * resolution_;
    if (!sdf_map_->isInMap(next_pos))
      return Eigen::Vector2i(-1, -1);  // Invalid
    
    // Check if next cell is walkable
    if (!isWalkable(next, safety_mode))
      return Eigen::Vector2i(-1, -1);  // Blocked
    
    // If we reached the goal, this is a jump point
    if (next == goal || (abs(next(0) - goal(0)) <= 1 && abs(next(1) - goal(1)) <= 1))
      return next;
    
    // Check for forced neighbors (defines jump points)
    if (hasForcedNeighbor(next, dir, safety_mode))
      return next;
    
    // If diagonal, check cardinal directions for jump points
    if (dir(0) != 0 && dir(1) != 0) {
      // Check horizontal and vertical jumps (iterative)
      Eigen::Vector2i h_dir(dir(0), 0);
      Eigen::Vector2i v_dir(0, dir(1));
      
      Eigen::Vector2i h_current = next;
      int h_count = 0;
      while (h_count < 100) {
        Eigen::Vector2i h_next = h_current + h_dir;
        h_count++;
        
        Eigen::Vector2d h_pos;
        h_pos(0) = origin_(0) + (h_next(0) + 0.5) * resolution_;
        h_pos(1) = origin_(1) + (h_next(1) + 0.5) * resolution_;
        if (!sdf_map_->isInMap(h_pos) || !isWalkable(h_next, safety_mode))
          break;
        
        if (h_next == goal || (abs(h_next(0) - goal(0)) <= 1 && abs(h_next(1) - goal(1)) <= 1))
          return next;
        
        if (hasForcedNeighbor(h_next, h_dir, safety_mode))
          return next;
        
        h_current = h_next;
      }
      
      Eigen::Vector2i v_current = next;
      int v_count = 0;
      while (v_count < 100) {
        Eigen::Vector2i v_next = v_current + v_dir;
        v_count++;
        
        Eigen::Vector2d v_pos;
        v_pos(0) = origin_(0) + (v_next(0) + 0.5) * resolution_;
        v_pos(1) = origin_(1) + (v_next(1) + 0.5) * resolution_;
        if (!sdf_map_->isInMap(v_pos) || !isWalkable(v_next, safety_mode))
          break;
        
        if (v_next == goal || (abs(v_next(0) - goal(0)) <= 1 && abs(v_next(1) - goal(1)) <= 1))
          return next;
        
        if (hasForcedNeighbor(v_next, v_dir, safety_mode))
          return next;
        
        v_current = v_next;
      }
    }
    
    // Continue jumping in the same direction
    current = next;
  }
  
  return Eigen::Vector2i(-1, -1);  // Max distance reached
}
}  // namespace apexnav_planner
