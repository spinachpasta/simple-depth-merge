#![warn(clippy::pedantic)]
#![allow(clippy::wildcard_imports)]
use cv::core::Vec3b;
use cv::prelude::MatExprTraitConst;
use opencv as cv;
use opencv::{core::*, highgui, imgcodecs::*, imgproc::*, Result};
use std::collections::HashMap;
use nalgebra as na;

struct DepthView {
    rgb: Mat,
    depth: Mat,
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
        let mut depth = Mat::default();
        cvt_color(&depth_rgb, &mut depth, COLOR_BGR2GRAY, 0)?;
        DepthView::new(image, depth, features)
    }
    pub fn new(rgb: Mat, depth: Mat, features: HashMap<String, Point_<i32>>) -> Result<DepthView> {
        let image_size = rgb.size()?;
        let depth_size = depth.size()?;
        assert!(
            !(image_size.width != depth_size.width && image_size.height != depth_size.height),
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
    pub fn debug_depth(&self, depth: &Mat) -> Result<Mat> {
        let mut preview = (Mat::zeros(self.width, self.height, cv::core::CV_8UC3)?).to_mat()?;

        for y in 0..self.height - 1 {
            for x in 0..self.width - 1 {
                let d = depth.at_2d::<u8>(y, x)?;
                let c = self.rgb.at_2d::<Vec3b>(y, x)?;
                let z: i32 = i32::from(*d);
                let p = preview.at_2d_mut::<Vec3b>(y, z)?;
                *p = *c;
            }
        }
        for (fpname, fp) in &self.features {
            let y = fp.y;
            let x = fp.x;
            let d = depth.at_2d::<u8>(y, x)?;
            let z: i32 = i32::from(*d);

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

    // pub fn get_zmatrix(&self, other: &DepthView) -> Vec4f {
    //     let mut m = Vec4f::new(0.0, 0.0, 0.0, 0.0);
    // }

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

    pub fn get_calibrated(&self, other: &DepthView) -> Result<Mat> {
        let mut calibrated = (Mat::zeros(self.width, self.height, cv::core::CV_8UC1)?).to_mat()?;
        let mut matched_features: Vec<&str> = self.match_features(other);

        for y in 0..self.height - 1 {
            for x in 0..self.width - 1 {
                let mut weights: HashMap<String, f64> = HashMap::new();
                let mut translations: HashMap<String, f64> = HashMap::new();
                let mut total_weight = 0.0;

                for fpname in &matched_features {
                    let fp = self.features.get(*fpname).unwrap();
                    let fp1 = other.features.get(*fpname).unwrap();
                    let dsq = (fp.x - x) * (fp.x - x) + (fp.y - y) * (fp.y - y);
                    let weight = 1.0 / (1.0 + f64::from(dsq));
                    weights.insert((*fpname).to_string(), weight);

                    let fpz = self.depth.at_2d::<u8>(fp.y, fp.x)?;
                    translations
                        .insert((*fpname).to_string(), (f64::from(fp1.x) - f64::from(*fpz)));
                    total_weight += weight;
                }
                let mut translation = 0.0;

                for fpname in &matched_features {
                    translation += translations.get(*fpname).unwrap()
                        * weights.get(*fpname).unwrap()
                        / total_weight;
                }
                let d = self.depth.at_2d::<u8>(y, x)?;
                let z: i32 = i32::from((*d)) + translation.floor() as i32;
                let zu8: u8 = z.clamp(0, 255) as u8;
                let p = calibrated.at_2d_mut::<u8>(y, x)?;
                *p = zu8;
            }
        }
        Ok(calibrated)
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

    highgui::named_window("pseudo right", 0)?;
    highgui::imshow("pseudo right", &front.debug_side()?)?;

    highgui::named_window("front", 0)?;
    highgui::imshow("front", &front.debug_features()?)?;

    highgui::named_window("side", 0)?;
    highgui::imshow("side", &side.debug_features()?)?;

    highgui::named_window("corrected", 0)?;
    highgui::imshow(
        "corrected",
        &front.debug_depth(&front.get_calibrated(&side)?)?,
    )?;

    highgui::wait_key(10000)?;
    Ok(())
}
