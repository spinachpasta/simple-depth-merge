

use nalgebra as na;
use opencv::{core::*, imgcodecs::*, imgproc::*, Result};

use std::collections::HashMap;
pub struct DepthView {
    pub rgb: Mat,
    pub depth: na::DMatrix<f64>,
    pub width: i32,
    pub height: i32,
    pub features: HashMap<String, Point_<i32>>,
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
            !(image_size.width != i32::try_from(depth_size.1).unwrap()
                && image_size.height != i32::try_from(depth_size.0).unwrap()),
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

    pub fn get_zmatrix(
        &self,
        other: &DepthView,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Const<3>,
        nalgebra::Const<1>,
        nalgebra::ArrayStorage<f64, 3, 1>,
    > {
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

    pub fn calibrate_z_linear(
        &self,
        other: &DepthView,
    ) -> nalgebra::Matrix<
        f64,
        nalgebra::Dynamic,
        nalgebra::Dynamic,
        nalgebra::VecStorage<f64, nalgebra::Dynamic, nalgebra::Dynamic>,
    > {
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
}