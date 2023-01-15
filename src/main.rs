#![warn(clippy::pedantic)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

use nalgebra as na;
use opencv::{core::*, highgui, viz, viz::Viz3dTrait, Result};

use std::collections::HashMap;
mod pointcloud;
mod depth_view;
use pointcloud::PointCloud;
use depth_view::DepthView;

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
    // highgui::named_window("front", 0)?;
    // highgui::imshow("front", &front.debug_features()?)?;

    // highgui::named_window("side", 0)?;
    // highgui::imshow("side", &side.debug_features()?)?;

    let side_affine = {
        let matrix = na::Matrix4::new(
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 1.0,
        );
        na::Affine3::<f64>::from_matrix_unchecked(matrix)
    };
    let mut cloud_side = PointCloud::new(&side.rgb, &side.calibrate_z_linear(&front), side_affine)?;

    let mut cloud_front = PointCloud::new(&front.rgb, &front.calibrate_z_linear(&side), na::Affine3::<f64>::identity())?;


    let mut viewer: viz::Viz3d = viz::Viz3d::new("side view")?;


    cloud_front.approximate_to(&cloud_side, &0.5);
    // cloud_side.approximate_to(&cloud_front, &0.5);

    // viewer.show_widget("side", &cloud_side.get_cv2_pointcloud()?.into(), Affine3d::default())?;
    viewer.show_widget("front", &cloud_front.get_cv2_pointcloud()?.into(), Affine3d::default())?;
    viewer.spin()?;

    highgui::wait_key(0)?;
    Ok(())
}
