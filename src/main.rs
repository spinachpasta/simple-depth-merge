#![warn(clippy::pedantic)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
use cv::core::Vec3b;
use cv::prelude::MatExprTraitConst;

use cv::types::{VectorOfPoint3d, VectorOfPoint3i};
use nalgebra as na;
use opencv as cv;
use opencv::{core::*, highgui, imgcodecs::*, imgproc::*, viz, viz::Viz3dTrait, Result};

use std::collections::HashMap;
struct DepthView {
    rgb: Mat,
    depth: na::DMatrix<f64>,
    width: i32,
    height: i32,
    features: HashMap<String, Point_<i32>>,
}

impl DepthView {
    pub fn from_filename(
        rgb_path: &str,
        depth_path: &str,
        features: HashMap<String, Point_<i32>>,
    ) -> Result<DepthView> {
        let image = imread(rgb_path, IMREAD_COLOR)?;
        let depth_rgb = imread(depth_path, IMREAD_COLOR)?;
        let mut depthgray = Mat::default();
        cvt_color(&depth_rgb, &mut depthgray, COLOR_BGR2GRAY, 0)?;
        let depth_size = depthgray.size()?;
        let depthu8 = na::DMatrix::<u8>::from_row_slice(
            depth_size.height as usize,
            depth_size.width as usize,
            depthgray.data_typed()?,
        );
        let depthf64 = na::convert::<na::DMatrix<u8>, na::DMatrix<f64>>(depthu8);

        DepthView::new(image, depthf64, features)
    }
    pub fn new(
        rgb: Mat,
        depth: na::DMatrix<f64>,
        features: HashMap<String, Point_<i32>>,
    ) -> Result<DepthView> {
        let image_size = rgb.size()?;
        let depth_size = depth.shape();
        assert!(
            !(image_size.width != i32::try_from(depth_size.1).unwrap() && image_size.height != i32::try_from(depth_size.0).unwrap() ),
            "size of rgb and depth image have to be equal"
        );
        let ret = DepthView {
            rgb,
            depth,
            width: image_size.width,
            height: image_size.height,
            features,
        };
        Ok(ret)
    }

    pub fn debug_features(&self) -> Result<Mat> {
        let mut preview = self.rgb.try_clone()?;

        for (fpname, fp) in &self.features {
            circle(
                &mut preview,
                *fp,
                5,
                Scalar_::new(255.0, 0.0, 0.0, 1.0),
                -1,
                1,
                0,
            )?;
            put_text(
                &mut preview,
                fpname,
                Point2i::new(fp.x, fp.y),
                FONT_HERSHEY_PLAIN,
                1.0,
                Scalar_::new(255.0, 0.0, 0.0, 1.0),
                1,
                1,
                false,
            )?;
        }
        Ok(preview)
    }
    pub fn get_depth(&self, x: i32, y: i32) -> f64 {
        self.depth[(y as usize, x as usize)]
    }
    pub fn debug_depth(&self, depth: &na::DMatrix<f64>) -> Result<Mat> {
        let mut preview = (Mat::zeros(self.width, self.height, cv::core::CV_8UC3)?).to_mat()?;

        for y in 0..self.height - 1 {
            for x in 0..self.width - 1 {
                let c = self.rgb.at_2d::<Vec3b>(y, x)?;
                let z = depth[(y as usize, x as usize)].round() as i32;
                if z < 0 || self.width <= z {
                    continue;
                }
                let p = preview.at_2d_mut::<Vec3b>(y, z)?;
                *p = *c;
            }
        }
        for (fpname, fp) in &self.features {
            let y = fp.y;
            let x = fp.x;
            let z = depth[(y as usize, x as usize)].round() as i32;

            circle(
                &mut preview,
                Point_ { x: z, y },
                5,
                Scalar_::new(255.0, 0.0, 0.0, 1.0),
                -1,
                1,
                0,
            )?;
            put_text(
                &mut preview,
                fpname,
                Point_ { x: z, y },
                FONT_HERSHEY_DUPLEX,
                1.0,
                Scalar_::new(255.0, 128.0, 128.0, 1.0),
                1,
                1,
                false,
            )?;
        }
        Ok(preview)
    }

    pub fn get_zmatrix(&self, other: &DepthView) -> nalgebra::Matrix<f64, nalgebra::Const<3>, nalgebra::Const<1>, nalgebra::ArrayStorage<f64, 3, 1>> {
        // let mut m: na::Matrix1x4<f64> = na::Matrix1x4::zeros();
        // m.set_row(0, na::RowVector4())
        let fpnames = self.match_features(other);

        let mut x = na::OMatrix::<f64, na::Dynamic, na::U2>::zeros(fpnames.len());
        let mut y = na::OVector::<f64, na::Dynamic>::zeros(fpnames.len());

        for (index, fpname) in fpnames.into_iter().enumerate() {
            let origin = self.features.get(fpname).unwrap();
            let target = other.features.get(fpname).unwrap();
            x[(index, 0)] = self.get_depth(origin.x, origin.y);
            x[(index, 1)] = f64::from(origin.y);
            y[(index, 0)] = f64::from(target.x);
        }
        let x_mean = x.row_mean();
        let y_mean = y.row_mean();

        let xx = {
            let subxmean = -x_mean;
            //subtract x
            let mut xx1 = x.clone();
            for (mut column, coeff) in xx1.column_iter_mut().zip(subxmean.iter()) {
                column.add_scalar_mut(*coeff);
            }
            xx1
        };
        let yy = {
            let subymean = -y_mean;
            //subtract y
            let mut yy1 = y.clone();
            for (mut column, coeff) in yy1.column_iter_mut().zip(subymean.iter()) {
                column.add_scalar_mut(*coeff);
            }
            yy1
        };

        println!("{xx}");
        println!("{yy}");
        // let m = (xx.clone() * xx.clone().transpose()).try_inverse().unwrap()*xx.clone()*yy.clone();
        let m = lstsq::lstsq(&xx, &yy, 1e-14).unwrap().solution;
        let w = y_mean - x_mean * m;
        
        na::Matrix3x1::<f64>::new(m.x, m.y, w.x)
    }

    pub fn calibrate_z_linear(&self, other: &DepthView) -> nalgebra::Matrix<f64, nalgebra::Dynamic, nalgebra::Dynamic, nalgebra::VecStorage<f64, nalgebra::Dynamic, nalgebra::Dynamic>> {
        let mut calibrated = na::DMatrix::<f64>::zeros(self.height as usize, self.width as usize);

        let m = self.get_zmatrix(other);
        println!("!{m}");
        for y in 0..self.height - 1 {
            for x in 0..self.width - 1 {
                // println!("{x} {y}");
                let d = na::Matrix1x3::<f64>::new(self.get_depth(x, y), f64::from(y), 1.0) * m;
                calibrated[(y as usize, x as usize)] = d[(0, 0)];
            }
        }
        let cmax = calibrated.max();
        let cmin = calibrated.min();
        println!("min {cmin} max {cmax}");
        calibrated
    }

    pub fn match_features(&self, other: &DepthView) -> Vec<&str> {
        let mut matched_features: Vec<&str> = Vec::<&str>::new();
        for fpname in self.features.keys() {
            for fpname1 in other.features.keys() {
                if *fpname == *fpname1 {
                    matched_features.push(fpname);
                    break;
                }
            }
        }
        matched_features
    }
    fn get_cv2_pointcloud(&self, depth: &na::DMatrix<f64>) -> Result<viz::WCloud> {
        let mut points = Vec::<Point3d>::new();
        let mut colors = Vec::<Point3i>::new();
        for y in 0..self.height - 1 {
            for x in 0..self.width - 1 {
                let z = depth[(y as usize, x as usize)];
                let p = Point3d {
                    x: f64::from(x),
                    y: f64::from(y),
                    z,
                };
                points.push(p);
                let color = self.rgb.at_2d::<Vec3b>(y, x)?;
                let color = Point3i {
                    x: i32::from(color.0[0]),
                    y: i32::from(color.0[1]),
                    z: i32::from(color.0[2]),
                };
                colors.push(color);
            }
        }
        let points = VectorOfPoint3d::from(points);
        let colors = VectorOfPoint3i::from(colors);
        viz::WCloud::new(&points, &colors)
    }
}

fn main() -> Result<()> {
    let mut feature_front = HashMap::new();
    feature_front.insert("nose".to_string(), Point_::<i32>::new(231, 290));
    feature_front.insert("eye".to_string(), Point_::<i32>::new(315, 210));
    feature_front.insert("chin".to_string(), Point_::<i32>::new(251, 363));
    feature_front.insert("hair".to_string(), Point_::<i32>::new(252, 130));

    let mut feature_side = HashMap::new();
    feature_side.insert("nose".to_string(), Point_::<i32>::new(178, 277));
    feature_side.insert("eye".to_string(), Point_::<i32>::new(280, 222));
    feature_side.insert("chin".to_string(), Point_::<i32>::new(219, 357));
    feature_side.insert("hair".to_string(), Point_::<i32>::new(217, 134));

    let front = DepthView::from_filename(
        "input/rgb/front.png",
        "input/depth/front.png",
        feature_front,
    )?;

    let side =
        DepthView::from_filename("input/rgb/side.png", "input/depth/side.png", feature_side)?;

    // highgui::named_window("pseudo right", 0)?;
    // highgui::imshow("pseudo right", &front.debug_side()?)?;

    // highgui::named_window("front", 0)?;
    // highgui::imshow("front", &front.debug_features()?)?;

    highgui::named_window("side", 0)?;
    highgui::imshow("side", &side.debug_features()?)?;

    highgui::named_window("corrected", 0)?;
    highgui::imshow(
        "corrected",
        &side.debug_depth(&side.calibrate_z_linear(&front))?,
    )?;

    let mut viewer: viz::Viz3d = viz::Viz3d::new("3dview")?;
    let cloud = side.get_cv2_pointcloud(&side.calibrate_z_linear(&front))?;

    viewer.show_widget("pointcloud", &cloud.into(), Affine3d::default())?;

    highgui::wait_key(0)?;
    Ok(())
}
