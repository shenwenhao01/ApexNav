// RiskMap2D: lightweight 2D risk layer stored on the same grid as SDFMap2D
#ifndef _RISK_MAP2D_H_
#define _RISK_MAP2D_H_

#include <Eigen/Eigen>
#include <vector>

namespace apexnav_planner {

class SDFMap2D;  // forward declaration to avoid circular include

class RiskMap2D {
public:
  explicit RiskMap2D(SDFMap2D* sdf_map);
  ~RiskMap2D() = default;

  // Allocate/resize internal buffer based on SDF map size; clears to zeros
  void resizeFromSDF();
  void reset(double value = 0.0);

  // Get/set by grid index or world position
  double get(const Eigen::Vector2i& idx) const;
  double get(const Eigen::Vector2d& pos) const;
  void set(const Eigen::Vector2i& idx, double value);
  void setMax(const Eigen::Vector2i& idx, double value);

  // Morphological maximum within a given radius (meters)
  void inflate(double radius_m);

private:
  SDFMap2D* sdf_map_;               // not owned
  std::vector<double> risk_buffer_; // [0, +inf) typical range [0,1]
};

}  // namespace apexnav_planner

#endif

