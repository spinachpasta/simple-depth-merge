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
        let mut preview = Mat::default();
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
        let image_size = self.rgb.size()?;
        let depth_size = self.depth.size()?;
        let mut preview =
            (Mat::zeros(image_size.width, image_size.height, cv::core::CV_8UC3)?).to_mat()?;

        for y in 0..image_size.height - 1 {
            for x in 0..image_size.width - 1 {
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

    let image = imread("input/rgb/front.png", IMREAD_COLOR)?;
    let depth_rgb = imread("input/depth/front.png", IMREAD_COLOR)?;
    let mut depth = Mat::default();
    cvt_color(&depth_rgb, &mut depth, COLOR_BGR2GRAY, 0)?;

    let image_size = image.size()?;
    let depth_size = depth.size()?;

    if image_size.width != depth_size.width && image_size.height != depth_size.height {
        panic!("size of rgb and depth image have to be equal");
    }

    let mut preview =
        (Mat::zeros(image_size.width, image_size.height, cv::core::CV_8UC3)?).to_mat()?;

    let x = 0;
    let y = 0;
    for y in 0..image_size.height - 1 {
        for x in 0..image_size.width - 1 {
            let d = depth.at_2d::<u8>(y, x)?;
            let c = image.at_2d::<Vec3b>(y, x)?;

            let mut z: i32 = -(*d as i32) + 256;
            if z < 0 {
                z = 0;
            }
            if z >= image_size.width {
                z = image_size.width - 1;
                continue;
            }
            let mut p = preview.at_2d_mut::<Vec3b>(y, z)?;
            *p = *c;
        }
    }

    let mut preview_front = Mat::default();
    image.copy_to(&mut preview_front)?;
    for fp in feature_front {
        circle(
            &mut preview_front,
            fp,
            5,
            Scalar_::new(255.0, 0.0, 0.0, 1.0),
            -1,
            1,
            0,
        )?;
    }

    highgui::named_window("pseudo right", 0)?;
    highgui::imshow("pseudo right", &preview)?;

    highgui::named_window("front", 0)?;
    highgui::imshow("front", &preview_front)?;

    highgui::wait_key(10000)?;
    Ok(())
}
