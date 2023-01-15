use cv::core::Vec3b;

use nalgebra as na;
use nalgebra::Affine3;
use opencv as cv;
use opencv::{core::*, viz, Result};

pub struct PointCloud {
    pub colors: Vec<na::Vector3<u8>>,
    pub points: Vec<na::OPoint<f64, na::Const<3>>>,
    pub transform: na::Affine3<f64>,
}

impl PointCloud {
    pub fn new(
        rgb_img: &Mat,
        depth: &na::DMatrix<f64>,
        transform: na::Affine3<f64>,
    ) -> Result<Self, cv::Error> {
        //TODO: calculate normal here
        let rgb_size = rgb_img.size()?;
        let width = rgb_size.width;
        let height = rgb_size.height;
        let mut points = Vec::<na::OPoint<f64, na::Const<3>>>::new();
        let mut colors = Vec::<na::Vector3<u8>>::new();

        for y in 0..height {
            for x in 0..width {
                // let idx = y * width + x;
                {
                    let z = depth[(y as usize, x as usize)];
                    let p = na::OPoint::<f64, na::Const<3>>::new(x.into(), y.into(), z);
                    points.push(p);
                }
                {
                    let color = rgb_img.at_2d::<Vec3b>(y, x)?;
                    colors.push(na::Vector3::<u8>::new(color.0[0], color.0[1], color.0[2]));
                }
            }
        }

        Ok(PointCloud {
            colors,
            points,
            transform,
        })
    }

    pub fn affine_applied(
        &self,
        transform: na::Affine3<f64>,
    ) -> Vec<na::OPoint<f64, na::Const<3>>> {
        let mut points = Vec::<na::OPoint<f64, na::Const<3>>>::new();
        for p in &self.points {
            let global_point = transform.transform_point(p);
            points.push(global_point);
        }
        points
    }
    pub fn get_global_points(&self) -> Vec<na::OPoint<f64, na::Const<3>>> {
        self.affine_applied(self.transform)
    }

    pub fn get_cv2_pointcloud(&self) -> Result<viz::WCloud> {
        let mut points = Mat::default();
        // let mut colors = Vec::<Vec3b>::new();
        let mut colors = Mat::default();
        unsafe {
            points.create_rows_cols(self.points.len().try_into().unwrap(), 1, CV_64FC3)?;
            colors.create_rows_cols(self.colors.len().try_into().unwrap(), 1, CV_8UC3)?;
        }
        let global_points = self.get_global_points();
        for idx in 0..self.points.len() {
            {
                let p = points.at_2d_mut::<Vec3d>(idx as i32, 0)?;
                let v = global_points[idx];
                p.0[0] = v.x;
                p.0[1] = v.y;
                p.0[2] = v.z;
            }
            {
                let p = colors.at_2d_mut::<Vec3b>(idx as i32, 0)?;
                p.0[0] = self.colors[idx].x;
                p.0[1] = self.colors[idx].y;
                p.0[2] = self.colors[idx].z;
            }
        }
        viz::WCloud::new(&points, &colors)
    }

    pub fn approximate_to(&mut self, other: &PointCloud, t: &f64) {
        let other_global = other.get_global_points();
        let mut other_local = Vec::<na::OPoint<f64, na::Const<3>>>::new();

        println!("{0}", self.transform.matrix());
        for p in other_global {
            other_local.push(self.transform.inverse_transform_point(&p));
        }
        let kdtree = kd_tree::KdTree::build_by_ordered_float(other_local);

        // let mut counter = 0;
        for p in &mut self.points {
            let p1 = kdtree.nearest(p).unwrap().item;
            p.z = p.z * (1.0 - t) + t * p1.z;
            // println!("{0}/{1}", counter, 512 * 512);
            // counter += 1;
        }
    }
}
