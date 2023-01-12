#![warn(clippy::pedantic)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
use cv::core::Vec3b;

use nalgebra as na;
use opencv as cv;
use opencv::{core::*, highgui, imgcodecs::*, imgproc::*, viz, viz::Viz3dTrait, Result};

use std::collections::HashMap;
use std::f64::consts::PI;
struct DepthView {
    rgb: Mat,
    depth: na::DMatrix<f64>,
    width: i32,
    height: i32,
    features: HashMap<String, Point_<i32>>,
}

struct PointCloud {
    colors: Vec<na::Vector3<u8>>,
    points: Vec<na::OPoint<f64, na::Const<3>>>,
    transform: na::Affine3<f64>,
}

impl PointCloud {
    pub fn new(
        rgb_img: &Mat,
        depth: &na::DMatrix<f64>,
        transform: na::Affine3<f64>,
    ) -> Result<Self, cv::Error> {
        //TODO: calculate normal here
        let rgb_size = rgb_img.size()?;
        let width = rgb_size.width;
        let height = rgb_size.height;
        let mut points = Vec::<na::OPoint<f64, na::Const<3>>>::new();
        let mut colors = Vec::<na::Vector3<u8>>::new();

        for y in 0..height {
            for x in 0..width {
                // let idx = y * width + x;
                {
                    let z = depth[(y as usize, x as usize)];
                    let p = na::OPoint::<f64, na::Const<3>>::new(x.into(), y.into(), z);
                    points.push(p);
                }
                {
                    let color = rgb_img.at_2d::<Vec3b>(y, x)?;
                    colors.push(na::Vector3::<u8>::new(color.0[0], color.0[1], color.0[2]));
                }
            }
        }

        Ok(PointCloud {
            points,
            colors,
            transform,
        })
    }

    fn get_cv2_pointcloud(&self) -> Result<viz::WCloud> {
        let mut points = Mat::default();
        // let mut colors = Vec::<Vec3b>::new();
        let mut colors = Mat::default();
        unsafe {
            points.create_rows_cols(self.points.len().try_into().unwrap(), 1, CV_64FC3)?;
            colors.create_rows_cols(self.colors.len().try_into().unwrap(), 1, CV_8UC3)?;
        }
        for idx in 0..self.points.len() {
            {
                let p = points.at_2d_mut::<Vec3d>(idx as i32, 0)?;
                let v = self.points[idx];
                let v = self.transform.transform_point(&v);
                p.0[0] = v.x;
                p.0[1] = v.y;
                p.0[2] = v.z;
            }
            {
                let p = colors.at_2d_mut::<Vec3b>(idx as i32, 0)?;
                p.0[0] = self.colors[idx].x;
                p.0[1] = self.colors[idx].y;
                p.0[2] = self.colors[idx].z;
            }
        }
        viz::WCloud::new(&points, &colors)
    }
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
    fn get_cv2_pointcloud(&self, depth: &na::DMatrix<f64>) -> Result<viz::WCloud> {
        let mut points = Mat::default();
        // let mut colors = Vec::<Vec3b>::new();
        let mut colors = Mat::default();
        unsafe {
            points.create_rows_cols(self.width * self.height, 1, CV_64FC3)?;

            colors.create_rows_cols(self.width * self.height, 1, CV_8UC3)?;
        }
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                {
                    let z = depth[(y as usize, x as usize)];
                    // let p = Vec3d:: (x,y,z);
                    let p = points.at_2d_mut::<Vec3d>(idx, 0)?;
                    p.0[0] = f64::from(x);
                    p.0[1] = f64::from(y);
                    p.0[2] = z;
                }
                {
                    let color = self.rgb.at_2d::<Vec3b>(y, x)?;
                    let p = colors.at_2d_mut::<Vec3b>(idx, 0)?;
                    *p = *color;
                }
            }
        }
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
    // highgui::named_window("front", 0)?;
    // highgui::imshow("front", &front.debug_features()?)?;

    // highgui::named_window("side", 0)?;
    // highgui::imshow("side", &side.debug_features()?)?;

    let side_affine = {
        let matrix = na::Matrix4::new(
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0,
        );
        na::Affine3::<f64>::from_matrix_unchecked(matrix)
    };
    let cloud_side = PointCloud::new(&side.rgb, &side.calibrate_z_linear(&front), side_affine)?;

    let cloud_front = PointCloud::new(&front.rgb, &front.calibrate_z_linear(&side), na::Affine3::<f64>::identity())?;


    let mut viewer: viz::Viz3d = viz::Viz3d::new("side view")?;
    viewer.show_widget("side", &cloud_side.get_cv2_pointcloud()?.into(), Affine3d::default())?;

    viewer.show_widget("front", &cloud_front.get_cv2_pointcloud()?.into(), Affine3d::default())?;
    viewer.spin()?;

    highgui::wait_key(0)?;
    Ok(())
}
