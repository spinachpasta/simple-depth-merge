use cv::core::{Vec3b, CV_8U};
use cv::prelude::MatExprTraitConst;
use ndarray::ArrayView3;
use opencv as cv;
use opencv::{highgui, imgcodecs::*, imgproc::*, prelude::*, Result};

fn main() -> Result<()> {
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

    highgui::named_window("hello opencv!", 0)?;
    highgui::imshow("hello opencv!", &preview)?;
    highgui::wait_key(10000)?;
    Ok(())
}
