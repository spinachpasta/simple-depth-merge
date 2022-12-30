use cv::core::{Vec3b, CV_8U};
use cv::prelude::MatExprTraitConst;
use ndarray::ArrayView3;
use opencv as cv;
use opencv::{core::*, highgui, imgcodecs::*, imgproc::*, imgproc::*, prelude::*, Result};
use std::collections::HashMap;

struct DepthView {
    rgb: Mat,
    depth: Mat,
    width: i32,
    height: i32,
    features: HashMap<String, Point_<i32>>,
}

impl DepthView {
    pub fn from_filename(
        rgb_path: String,
        depth_path: String,
        features: HashMap<String, Point_<i32>>,
    ) -> Result<DepthView> {
        let image = imread(&rgb_path, IMREAD_COLOR)?;
        let depth_rgb = imread(&depth_path, IMREAD_COLOR)?;
        let mut depth = Mat::default();
        cvt_color(&depth_rgb, &mut depth, COLOR_BGR2GRAY, 0)?;
        Ok(DepthView::new(image, depth, features)?)
    }
    pub fn new(rgb: Mat, depth: Mat, features: HashMap<String, Point_<i32>>) -> Result<DepthView> {
        let image_size = rgb.size()?;
        let depth_size = depth.size()?;
        if image_size.width != depth_size.width && image_size.height != depth_size.height {
            panic!("size of rgb and depth image have to be equal");
        }
        let ret = DepthView {
            rgb: rgb,
            depth: depth,
            width: image_size.width,
            height: image_size.height,
            features: features,
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
        let mut preview =
            (Mat::zeros(self.width, self.height, cv::core::CV_8UC3)?).to_mat()?;

        for y in 0..self.height - 1 {
            for x in 0..self.width - 1 {
                let d = self.depth.at_2d::<u8>(y, x)?;
                let c = self.rgb.at_2d::<Vec3b>(y, x)?;
                let z: i32 = *d as i32;
                let mut p = preview.at_2d_mut::<Vec3b>(y, z)?;
                *p = *c;
            }
        }
        Ok(preview)
    }
}

fn main() -> Result<()> {
    let feature_front: [Point_<i32>; 4] = [
        Point_::<i32>::new(231, 290),
        Point_::<i32>::new(315, 210),
        Point_::<i32>::new(251, 363),
        Point_::<i32>::new(252, 130),
    ];
    let feature_side: [Point_<i32>; 3] = [
        Point_::<i32>::new(1, 1),
        Point_::<i32>::new(1, 1),
        Point_::<i32>::new(1, 1),
    ];

    let mut feature_front = HashMap::new();
    feature_front.insert("nose".to_string(), Point_::<i32>::new(231, 290));
    feature_front.insert("eye".to_string(), Point_::<i32>::new(315, 210));
    feature_front.insert("chin".to_string(), Point_::<i32>::new(251, 363));
    feature_front.insert("hair".to_string(), Point_::<i32>::new(252, 130));
    
    let front = DepthView::from_filename(
        "input/rgb/front.png".to_string(),
        "input/depth/front.png".to_string(),
        feature_front
    )?;

    highgui::named_window("pseudo right", 0)?;
    highgui::imshow("pseudo right", &front.debug_side()?)?;

    highgui::named_window("front", 0)?;
    highgui::imshow("front", &front.debug_features()?)?;

    highgui::wait_key(10000)?;
    Ok(())
}
