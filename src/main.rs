#![warn(clippy::pedantic)]
#![allow(clippy::wildcard_imports)]
use cv::core::Vec3b;
use cv::prelude::MatExprTraitConst;

use na::Matrix1x4;
use nalgebra as na;
use opencv as cv;
use opencv::{core::*, highgui, imgcodecs::*, imgproc::*, Result};
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
            !(image_size.width != depth_size.1 as i32 && image_size.height != depth_size.0 as i32),
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
    pub fn debug_side(&self) -> Result<Mat> {
        self.debug_depth(&self.depth)
    }
    pub fn get_depth(&self, x: &i32, y: &i32) -> f64 {
        self.depth[(*y as usize, *x as usize)]
    }
    pub fn get_depth_round(&self, x: &i32, y: &i32) -> i32 {
        let r = self.get_depth(x, y).round();
        let d = r as i32;
        d
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

    pub fn get_zmatrix(&self, other: &DepthView) -> Result<na::Matrix4x1<f64>> {
        // let mut m: na::Matrix1x4<f64> = na::Matrix1x4::zeros();
        // m.set_row(0, na::RowVector4())
        let fpnames = self.match_features(other);

        let mut x = na::OMatrix::<f64, na::Dynamic, na::U3>::zeros(fpnames.len());
        let mut y = na::OVector::<f64, na::Dynamic>::zeros(fpnames.len());
        let mut index: usize = 0;

        for (index, fpname) in fpnames.into_iter().enumerate() {
            let origin = self.features.get(fpname).unwrap();
            let target = other.features.get(fpname).unwrap();
            x[(index, 0)] = f64::from(origin.x);
            x[(index, 1)] = f64::from(origin.y);
            x[(index, 2)] = self.get_depth(&origin.x, &origin.y);
            y[(index, 0)] = f64::from(target.x);
        }
        let xmean = x.row_mean();
        let ymean = y.row_mean();
        //subtract x
        let mut xx = x.clone();
        for (mut column, coeff) in xx.column_iter_mut().zip(xmean.iter()) {
            column.add_scalar_mut(*coeff)
        }
        //subtract y
        let mut yy = y.clone();
        for (mut column, coeff) in yy.column_iter_mut().zip(ymean.iter()) {
            column.add_scalar_mut(*coeff)
        }

        println!("f{xx} {yy}");
        let m = lstsq::lstsq(&xx, &yy, 1e-14).unwrap().solution;
        let w = ymean - xmean * m;
        let m1 = na::Matrix4x1::<f64>::new(m.x, m.y, m.z, w.x);
        Ok(m1)
    }

    pub fn calibrate_z_linear(&self, other: &DepthView) -> Result<na::DMatrix<f64>> {
        let mut calibrated = na::DMatrix::<f64>::zeros(self.height as usize, self.width as usize);

        let m = self.get_zmatrix(other)?;
        println!("!{m}");
        for y in 0..self.height - 1 {
            for x in 0..self.width - 1 {
                // println!("{x} {y}");
                let d =
                    (Matrix1x4::<f64>::new(0.0, y as f64, self.get_depth(&x, &y), 1.0) * m);
                calibrated[(y as usize, x as usize)] = d[(0, 0)];
            }
        }
        Ok(calibrated)
    }

    pub fn match_features(&self, other: &DepthView) -> Vec<&str> {
        let mut matched_features: Vec<&str> = Vec::<&str>::new();
        for (fpname, fp) in &self.features {
            for (fpname1, fp1) in &other.features {
                if *fpname == *fpname1 {
                    matched_features.push(fpname);
                    break;
                }
            }
        }
        matched_features
    }
}

fn main() -> Result<()> {
    let mut feature_front = HashMap::new();
    feature_front.insert("nose".to_string(), Point_::<i32>::new(231, 290));
    feature_front.insert("eye".to_string(), Point_::<i32>::new(315, 210));
    feature_front.insert("chin".to_string(), Point_::<i32>::new(251, 363));
    // feature_front.insert("hair".to_string(), Point_::<i32>::new(252, 130));

    let mut feature_side = HashMap::new();
    feature_side.insert("nose".to_string(), Point_::<i32>::new(178, 277));
    feature_side.insert("eye".to_string(), Point_::<i32>::new(280, 222));
    feature_side.insert("chin".to_string(), Point_::<i32>::new(219, 357));
    // feature_side.insert("hair".to_string(), Point_::<i32>::new(217, 134));

    let front = DepthView::from_filename(
        "input/rgb/front.png",
        "input/depth/front.png",
        feature_front,
    )?;

    let side =
        DepthView::from_filename("input/rgb/side.png", "input/depth/side.png", feature_side)?;

    highgui::named_window("pseudo right", 0)?;
    highgui::imshow("pseudo right", &front.debug_side()?)?;

    // highgui::named_window("front", 0)?;
    // highgui::imshow("front", &front.debug_features()?)?;

    highgui::named_window("side", 0)?;
    highgui::imshow("side", &side.debug_features()?)?;

    highgui::named_window("corrected", 0)?;
    highgui::imshow(
        "corrected",
        &front.debug_depth(&front.calibrate_z_linear(&side)?)?,
    )?;

    highgui::wait_key(10000)?;
    Ok(())
}
