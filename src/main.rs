use cv::core::{CV_8U, Vec3b};
use cv::prelude::MatExprTraitConst;
use ndarray::ArrayView3;
use opencv as cv;
use opencv::{highgui, imgcodecs::*, prelude::*, Result};

fn main() -> Result<()> {
    let image = imread("input/rgb/front.png", IMREAD_COLOR)?;
    let depth = imread("input/depth/front.png", IMREAD_COLOR)?;

    let image_size = image.size()?;
    let depth_size = depth.size()?;

    if image_size.width != depth_size.width && image_size.height != depth_size.height {
        panic!("size of rgb and depth image have to be equal");
    }

    let mut preview =
        (Mat::zeros(image_size.width, image_size.height, cv::core::CV_8UC3)?).to_mat()?;

    let x = 0;
    let y = 0;
    // println!("{}", depth_pixels[(x + image_size.width * y) as usize]);
    for y in 0..image_size.height - 1 {
        for x in 0..image_size.width - 1 {
            // let d = depth.at_2d::<u8>(y, x)?;
            // let c = image.at_3d::<u8>(y, x, 0)?;
            let mut p = preview.at_2d_mut::<Vec3b>(y, x)?;
            *p = Vec3b::all(220);
        }
    }

    highgui::named_window("hello opencv!", 0)?;
    highgui::imshow("hello opencv!", &preview)?;
    highgui::wait_key(10000)?;
    Ok(())
}
