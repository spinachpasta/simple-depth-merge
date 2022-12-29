use opencv::{
    highgui,
    imgcodecs::{self, IMREAD_COLOR},
    prelude::MatTraitConstManual,
    Result,
};

fn main() -> Result<()> {
    let image = imgcodecs::imread("input/rgb/front.png", IMREAD_COLOR)?;
    let depth = imgcodecs::imread("input/depth/front.png", IMREAD_COLOR)?;

    let image_size = image.size()?;
    let depth_size = depth.size()?;

    if image_size.width != depth_size.width && image_size.height != depth_size.height {
        panic!("size of rgb and depth image have to be equal");
    }

    // let image_pixels = image.data_bytes()?;
    let depth_pixels = depth.data_bytes()?;

    let x = 10;
    let y = 10;
    println!("{}", depth_pixels[(x + image_size.width * y) as usize]);

    highgui::named_window("hello opencv!", 0)?;
    highgui::imshow("hello opencv!", &depth)?;
    highgui::wait_key(10000)?;
    Ok(())
}
