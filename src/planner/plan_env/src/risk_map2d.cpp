/**
 * @file risk_map2d.cpp
 * @brief 2D risk layer implementation on top of SDFMap2D grid
 */

 #include <algorithm>
 #include <plan_env/risk_map2d.h>
 #include <plan_env/sdf_map2d.h>
 
 namespace apexnav_planner {
 
 RiskMap2D::RiskMap2D(SDFMap2D* sdf_map) : sdf_map_(sdf_map)
 {
   resizeFromSDF();
 }
 
 void RiskMap2D::resizeFromSDF()
 {
   risk_buffer_.assign(sdf_map_->getVoxelNum(), 0.0);
 }
 
 void RiskMap2D::reset(double value)
 {
   std::fill(risk_buffer_.begin(), risk_buffer_.end(), value);
 }
 
 double RiskMap2D::get(const Eigen::Vector2i& idx) const
 {
   if (!sdf_map_->isInMap(idx))
     return 0.0;
   return risk_buffer_[sdf_map_->toAddress(idx)];
 }
 
 double RiskMap2D::get(const Eigen::Vector2d& pos) const
 {
   Eigen::Vector2i idx;
   sdf_map_->posToIndex(pos, idx);
   return get(idx);
 }
 
 void RiskMap2D::set(const Eigen::Vector2i& idx, double value)
 {
   if (!sdf_map_->isInMap(idx))
     return;
   risk_buffer_[sdf_map_->toAddress(idx)] = value;
 }
 
 void RiskMap2D::setMax(const Eigen::Vector2i& idx, double value)
 {
   if (!sdf_map_->isInMap(idx))
     return;
   auto adr = sdf_map_->toAddress(idx);
   risk_buffer_[adr] = std::max(risk_buffer_[adr], value);
 }
 
 void RiskMap2D::inflate(double radius_m)
 {
   if (radius_m <= 0.0)
     return;
 
   // Convert radius in meters to number of cells (at least 1 if >0)
   const int rad_cells = std::max(1, int(std::ceil(radius_m / sdf_map_->getResolution())));
 
   // Determine grid size by scanning addresses (via SDF map params)
   Eigen::Vector2d origin, size;
   sdf_map_->getRegion(origin, size);
 
   // Infer width/height from voxel indexing
   // We rely on SDFMap2D exposing address conversions and map_voxel_num_ bounds via isInMap.
   // Brute-force inflate: for each cell, take local max within disk radius.
   std::vector<double> inflated = risk_buffer_;  // copy
 
   // Determine bounds using indices (0..W-1, 0..H-1)
   // We probe by converting address 0 to idx (0,0) and increment; safer: iterate by index.
   // To get grid dims, use binary search on addressToIdx over range [0, N-1]
   // However, SDFMap2D provides map_voxel_num_ only internally. We'll reconstruct dims by walking.
 
   // Reconstruct grid dimensions by scanning until y wraps (expensive but done once per inflate)
   // We can safely assume a rectangular grid stored in row-major order with address = x*H + y.
   // So height H is the number of unique y values; find H by increasing address until x increments.
   const int N = static_cast<int>(risk_buffer_.size());
   // Estimate height by scanning y changes; fallback to square root if something goes wrong.
   int H = 0;
   {
     // Use the addressToIdx exposed by SDFMap2D
     Eigen::Vector2i last = sdf_map_->addressToIdx(0);
     for (int a = 1; a < N; ++a) {
       Eigen::Vector2i cur = sdf_map_->addressToIdx(a);
       if (cur.x() != last.x()) {
         H = a;  // first row length
         break;
       }
     }
     if (H <= 0) H = std::max(1, int(std::sqrt(double(N))));
   }
   const int W = (H > 0) ? (N / H) : 0;
 
   auto addr = [&](int x, int y) { return x * H + y; };
 
   for (int x = 0; x < W; ++x) {
     for (int y = 0; y < H; ++y) {
       double m = 0.0;
       for (int dx = -rad_cells; dx <= rad_cells; ++dx) {
         for (int dy = -rad_cells; dy <= rad_cells; ++dy) {
           if (dx * dx + dy * dy > rad_cells * rad_cells) continue;
           const int xx = x + dx;
           const int yy = y + dy;
           if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
           m = std::max(m, risk_buffer_[addr(xx, yy)]);
         }
       }
       inflated[addr(x, y)] = m;
     }
   }
 
   risk_buffer_.swap(inflated);
 }
 
 }  // namespace apexnav_planner
 
 