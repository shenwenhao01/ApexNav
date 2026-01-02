#include <path_searching/astar2d.h>
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

  // ARA* initialization
  best_end_node_ = nullptr;
  current_epsilon_ = 2.5;  // Initial epsilon for weighted A*
  epsilon_decrease_factor_ = 0.5;  // Decrease factor for epsilon
}

void Astar2D::reset()
{
  open_set_map_.clear();
  close_set_map_.clear();
  path_nodes_.clear();
  incons_set_.clear();

  std::priority_queue<Node2DPtr, std::vector<Node2DPtr>, NodeComparator2D> empty_queue;
  open_set_.swap(empty_queue);
  for (int i = 0; i < use_node_num_; i++) {
    path_node_pool_[i]->parent = nullptr;
  }
  use_node_num_ = 0;
  iter_num_ = 0;
  best_end_node_ = nullptr;
  current_epsilon_ = 2.5;  // Reset epsilon
}

void Astar2D::setResolution(const double& res)
{
  resolution_ = res;
  this->inv_resolution_ = 1.0 / resolution_;
}

int Astar2D::astarSearch(const Eigen::Vector2d& start_pt, const Eigen::Vector2d& end_pt,
    double success_dist, double max_time, int safety_mode)
{
  current_start_pt_ = start_pt;
  current_end_pt_ = end_pt;
  best_end_node_ = nullptr;
  path_nodes_.clear();

  const auto t1 = ros::Time::now();
  double epsilon = current_epsilon_;

  // First search: find initial solution
  // Use most of the time for first search (80%), save rest for improvements
  double first_search_time = max_time * 0.8;
  int result = improvePath(start_pt, end_pt, success_dist, first_search_time, safety_mode, epsilon);
  
  if (result == NO_PATH) {
    // If no path found in first search, check if we have a best node
    if (best_end_node_ != nullptr) {
      computePath(end_pt);
      return REACH_END;
    }
    return NO_PATH;
  }

  // If we found a solution, try to improve it within remaining time limit
  double remaining_time = max_time - (ros::Time::now() - t1).toSec();
  while (remaining_time > 0.001 && epsilon > 1.0 + 1e-6) {
    // Move inconsistent nodes from close set to open set
    for (auto& pair : incons_set_) {
      Node2DPtr node = pair.second;
      auto close_iter = close_set_map_.find(node->index);
      if (close_iter != close_set_map_.end()) {
        close_set_map_.erase(close_iter);
        open_set_map_[node->index] = node;
      }
    }
    incons_set_.clear();

    // Decrease epsilon
    epsilon = max(1.0, epsilon * epsilon_decrease_factor_);
    current_epsilon_ = epsilon;

    // CRITICAL: When epsilon changes, we must update ALL nodes in open_set
    // because f_score = g_score + epsilon * h, and h depends on epsilon
    // Rebuild the entire priority queue with updated f_scores
    std::priority_queue<Node2DPtr, std::vector<Node2DPtr>, NodeComparator2D> new_open_set;
    for (auto& pair : open_set_map_) {
      Node2DPtr node = pair.second;
      // Update f_score with new epsilon
      node->f_score = node->g_score + epsilon * lambda_heu_ * getDiagHeu(node->position, end_pt);
      new_open_set.push(node);
    }
    open_set_ = new_open_set;

    // Improve path with new epsilon, use all remaining time
    int improve_result = improvePath(start_pt, end_pt, success_dist, remaining_time, safety_mode, epsilon);
    
    // Update remaining time
    remaining_time = max_time - (ros::Time::now() - t1).toSec();
    
    // If time is up, break (even if we found a solution, we can't improve further)
    if (remaining_time <= 0.001) {
      break;
    }
    
    // If no improvement found and we already have a solution, we can break
    // (This happens when open_set becomes empty without finding a better solution)
    if (improve_result == NO_PATH && best_end_node_ != nullptr) {
      // Check if we can still improve (if open_set is empty, no more improvements possible)
      if (open_set_.empty() && incons_set_.empty()) {
        break;
      }
    }
  }

  // Compute final path
  if (best_end_node_ != nullptr) {
    computePath(end_pt);
    return REACH_END;
  }

  return NO_PATH;
}

int Astar2D::improvePath(const Eigen::Vector2d& start_pt, const Eigen::Vector2d& end_pt,
    double success_dist, double max_time, int safety_mode, double epsilon)
{
  Eigen::Vector2i end_index;
  posToIndex(end_pt, end_index);

  // Initialize start node if not already in open or close set
  Eigen::Vector2i start_idx;
  posToIndex(start_pt, start_idx);
  
  auto open_iter = open_set_map_.find(start_idx);
  auto close_iter = close_set_map_.find(start_idx);
  
  if (open_iter == open_set_map_.end() && close_iter == close_set_map_.end()) {
    // Start node not in any set, initialize it
    Node2DPtr start_node = path_node_pool_[0];
    start_node->parent = nullptr;
    start_node->position = start_pt;
    start_node->index = start_idx;
    start_node->g_score = 0.0;
    start_node->f_score = epsilon * lambda_heu_ * getDiagHeu(start_node->position, end_pt);
    open_set_.push(start_node);
    open_set_map_[start_node->index] = start_node;
    use_node_num_ = 1;
  } else if (open_iter == open_set_map_.end() && close_iter != close_set_map_.end()) {
    // Start node is in close set, move it to open set for re-expansion
    Node2DPtr start_node = close_iter->second;
    close_set_map_.erase(close_iter);
    start_node->g_score = 0.0;  // Reset g_score for start
    start_node->f_score = epsilon * lambda_heu_ * getDiagHeu(start_node->position, end_pt);
    start_node->parent = nullptr;
    open_set_.push(start_node);
    open_set_map_[start_idx] = start_node;
  } else if (open_iter != open_set_map_.end()) {
    // Start node is already in open set, ensure it has correct g_score
    Node2DPtr start_node = open_iter->second;
    if (start_node->g_score > 1e-6) {
      start_node->g_score = 0.0;
      start_node->parent = nullptr;
      start_node->f_score = epsilon * lambda_heu_ * getDiagHeu(start_node->position, end_pt);
      // LAZY UPDATE: Just push the updated node to the queue
      open_set_.push(start_node);
    }
  }

  const auto t1 = ros::Time::now();

  while (!open_set_.empty()) {
    // Check time limit
    if ((ros::Time::now() - t1).toSec() > max_time) {
      // Before returning, check if we have a solution in best_end_node_
      // Also check open_set for any node that reached the goal
      Node2DPtr best = getBestNode();
      if (best != nullptr) {
        early_terminate_cost_ = best->g_score + getDiagHeu(best->position, end_pt);
      }
      // If we have a solution, return REACH_END instead of NO_PATH
      if (best_end_node_ != nullptr) {
        return REACH_END;
      }
      return NO_PATH;
    }

    // LAZY UPDATE: Pop nodes until we find a valid one
    // Skip nodes that are no longer in open_set_map_ or have outdated f_scores
    // 
    // SAFETY NOTE: This lazy update strategy is safe because:
    // 1. priority_queue's heap structure is based on pointer positions, not values
    // 2. Modifying node->f_score after push doesn't corrupt the heap structure
    // 3. We skip outdated nodes during pop, so they never affect correctness
    // 4. The same node pointer may appear multiple times in the queue (old and new versions),
    //    but this is safe because we always check validity before using a node
    Node2DPtr cur_node = nullptr;
    while (!open_set_.empty()) {
      cur_node = open_set_.top();
      open_set_.pop();
      
      // Check if node is still in open_set_map_ (hasn't been closed or removed)
      auto iter = open_set_map_.find(cur_node->index);
      if (iter == open_set_map_.end() || iter->second != cur_node) {
        // Node has been removed or replaced, skip it (this is a "zombie" node from lazy update)
        continue;
      }
      
      // Check if f_score is still valid (node hasn't been updated)
      // This handles the case where we updated the node's f_score and re-pushed it
      double h_val = lambda_heu_ * getDiagHeu(cur_node->position, end_pt);
      double expected_f = cur_node->g_score + epsilon * h_val;
      if (fabs(cur_node->f_score - expected_f) > 1e-6) {
        // f_score is outdated, update it and re-insert
        // This creates a new entry in the queue with the updated f_score
        // The old entry (with outdated f_score) will be skipped in future pops
        cur_node->f_score = expected_f;
        open_set_.push(cur_node);
        continue;
      }
      
      // Node is valid, use it
      break;
    }
    
    // If we exhausted the queue without finding a valid node, we're done
    if (cur_node == nullptr || open_set_map_.find(cur_node->index) == open_set_map_.end()) {
      break;
    }
    
    // Check if reached goal
    bool reach_end = abs(cur_node->index(0) - end_index(0)) <= 1 && 
                     abs(cur_node->index(1) - end_index(1)) <= 1;
    if ((cur_node->position - end_pt).norm() < success_dist)
      reach_end = true;

    if (reach_end) {
      // Found a solution, update best_end_node if this is better
      if (best_end_node_ == nullptr || cur_node->g_score < best_end_node_->g_score) {
        best_end_node_ = cur_node;
      }
      // Close the goal node, but allow it to be re-opened if we find a better path
      // This allows ARA* to continue searching for better solutions
      // Note: cur_node was already popped in the lazy update loop above
      open_set_map_.erase(cur_node->index);
      close_set_map_[cur_node->index] = cur_node;
      iter_num_ += 1;
      // Continue searching - if we find a better path to goal, it will be in incons_set
      continue;
    }

    // Node is valid and consistent (already checked in lazy update above)
    // Remove it from open_set_map_ and add to close_set
    open_set_map_.erase(cur_node->index);
    close_set_map_[cur_node->index] = cur_node;
    iter_num_ += 1;

    Eigen::Vector2d cur_pos = cur_node->position;
    Eigen::Vector2d nbr_pos;

    std::vector<Eigen::Vector2d> steps = generateSteps(cur_pos);
    for (auto step : steps) {
      nbr_pos = cur_pos + step;

      // Skip safety raycast if still near start to avoid immediate termination
      if ((nbr_pos - start_pt).norm() > 0.25) {
        // Check safety
        if (!checkPointSafety(nbr_pos, safety_mode))
          continue;

        bool safe = true;
        Vector2d dir = nbr_pos - cur_pos;
        double len = dir.norm();
        dir.normalize();
        for (double l = 0.025; l < len; l += 0.025) {
          Vector2d ckpt = cur_pos + l * dir;
          if (!checkPointSafety(ckpt, safety_mode)) {
            safe = false;
            break;
          }
        }
        if (!safe)
          continue;
      }

      Eigen::Vector2i nbr_idx;
      posToIndex(nbr_pos, nbr_idx);

      // Check if neighbor is in close set
      auto close_iter = close_set_map_.find(nbr_idx);
      if (close_iter != close_set_map_.end()) {
        // Check if we found a better path to a closed node
        Node2DPtr closed_node = close_iter->second;
        double new_g = cur_node->g_score + step.norm();
        if (new_g < closed_node->g_score - 1e-6) {
          // Found better path, update the node and mark as inconsistent
          closed_node->g_score = new_g;
          closed_node->parent = cur_node;
          incons_set_[nbr_idx] = closed_node;
        }
        continue;
      }

      // Update or create neighbor node
      double tmp_g_score = step.norm() + cur_node->g_score;
      auto node_iter = open_set_map_.find(nbr_idx);
      
      Node2DPtr neighbor;
      if (node_iter == open_set_map_.end()) {
        // New node
        neighbor = path_node_pool_[use_node_num_];
        use_node_num_ += 1;
        if (use_node_num_ == allocate_num_) {
          cout << "run out of node pool." << endl;
          return NO_PATH;
        }
        neighbor->index = nbr_idx;
        neighbor->position = nbr_pos;
        neighbor->g_score = tmp_g_score;
        neighbor->f_score = tmp_g_score + epsilon * lambda_heu_ * getDiagHeu(nbr_pos, end_pt);
        neighbor->parent = cur_node;
        open_set_.push(neighbor);
        open_set_map_[nbr_idx] = neighbor;
      } else {
        // Existing node, update if better
        neighbor = node_iter->second;
        if (tmp_g_score < neighbor->g_score - 1e-6) {
          neighbor->g_score = tmp_g_score;
          neighbor->f_score = tmp_g_score + epsilon * lambda_heu_ * getDiagHeu(nbr_pos, end_pt);
          neighbor->parent = cur_node;
          // LAZY UPDATE: Just push the updated node to the queue
          // The old entry will be skipped during pop (lazy update handles it)
          open_set_.push(neighbor);
        }
      }
    }
  }

  // Open set is empty - check if we found a solution
  if (best_end_node_ != nullptr) {
    return REACH_END;
  }
  
  // No path found in this iteration
  return NO_PATH;
}

Node2DPtr Astar2D::getBestNode()
{
  Node2DPtr best = nullptr;
  double best_cost = 1e10;

  // Check open set
  std::priority_queue<Node2DPtr, std::vector<Node2DPtr>, NodeComparator2D> temp_queue = open_set_;
  while (!temp_queue.empty()) {
    Node2DPtr node = temp_queue.top();
    temp_queue.pop();
    if (best == nullptr || node->g_score < best_cost) {
      best = node;
      best_cost = node->g_score;
    }
  }

  // Check best_end_node
  if (best_end_node_ != nullptr) {
    if (best == nullptr || best_end_node_->g_score < best_cost) {
      best = best_end_node_;
    }
  }

  return best;
}

void Astar2D::computePath(const Eigen::Vector2d& end)
{
  if (best_end_node_ == nullptr) {
    path_nodes_.clear();
    return;
  }
  backtrack(best_end_node_, end);
}

void Astar2D::updateVertex(Node2DPtr node, const Eigen::Vector2d& end_pt, double epsilon)
{
  if (node->g_score + epsilon * lambda_heu_ * getDiagHeu(node->position, end_pt) < node->f_score - 1e-6) {
    node->f_score = node->g_score + epsilon * lambda_heu_ * getDiagHeu(node->position, end_pt);
    if (open_set_map_.find(node->index) != open_set_map_.end()) {
      // LAZY UPDATE: Just push the updated node to the queue
      open_set_.push(node);
    }
  }
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

double Astar2D::getEarlyTerminateCost()
{
  return early_terminate_cost_;
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
}  // namespace apexnav_planner
