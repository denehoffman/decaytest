use std::{collections::HashMap, fs::File, io::BufWriter, path::Path};

use factorial::Factorial;
use fastrand::Rng;
use fastrand_contrib::RngExt;
use itertools::Itertools;
use laddu::{
    prelude::{FourMomentum, FourVector, Vector3, Vector4},
    PI,
};

fn breakup_momentum(a: f64, b: f64, c: f64) -> f64 {
    let x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c);
    f64::sqrt(x) / (2.0 * a)
}

pub struct Decay {
    decaying_p4: Vector4<f64>,
    masses: Vec<f64>,
    max_weight: f64,
    budget: f64,
    n: usize,
}

impl Decay {
    pub fn new(decaying_p4: &Vector4<f64>, masses: &[f64]) -> Self {
        assert!(decaying_p4.m() > masses.iter().sum());
        Self {
            decaying_p4: *decaying_p4,
            masses: masses.to_vec(),
            max_weight: f64::powi(
                decaying_p4.m() - masses.iter().sum::<f64>(),
                (masses.len() - 2) as i32,
            ) * (PI * f64::powi(2.0 * PI, (masses.len() - 2) as i32)
                / (masses.len() - 2).factorial() as f64)
                / decaying_p4.m(),
            budget: decaying_p4.m() - masses.iter().sum::<f64>(),
            n: masses.len(),
        }
    }
    pub fn generate(&self, rng: &mut Rng) -> (Vec<Vector4<f64>>, f64) {
        // set up initial decay vectors and weight
        let mut dec_momenta = vec![Vector4::default(); self.n];
        let mut weight = self.max_weight;

        // generate sorted random capped by 0 and 1
        let mut r: Vec<f64> = vec![0.0; self.n];
        if self.n > 2 {
            r.iter_mut()
                .skip(1)
                .take(self.n - 1)
                .for_each(|rv| *rv = rng.f64());
            r.sort_by(|a, b| a.total_cmp(b));
        }
        r[self.n - 1] = 1.0;

        // generate intermediate masses
        let mut int_masses = vec![0.0; self.n];
        let mut child_masses = 0.0;
        for i in 0..self.n {
            child_masses += self.masses[i];
            int_masses[i] = r[i] * self.budget + child_masses;
        }

        // generate momenta magnitudes
        let mut q = vec![0.0; self.n];
        for i in 0..self.n - 1 {
            q[i] = breakup_momentum(int_masses[i + 1], int_masses[i], self.masses[i + 1]);
            weight *= q[i];
        }

        // assign momentum to particle 0
        dec_momenta[0] = Vector4::new(
            f64::sqrt(q[0].powi(2) + self.masses[0].powi(2)),
            0.0,
            0.0,
            q[0],
        );

        // assign momentum to other particles
        let mut i = 1;
        loop {
            dec_momenta[i] = Vector4::new(
                f64::sqrt(q[i - 1].powi(2) + self.masses[i].powi(2)),
                0.0,
                0.0,
                -q[i - 1],
            );

            // rotate by random spherical angles
            let cos_y = rng.f64_range(-1.0..1.0);
            let sin_y = f64::sqrt(1.0 - cos_y.powi(2));
            let phi = rng.f64_range(0.0..2.0 * PI);
            let cos_z = f64::cos(phi);
            let sin_z = f64::sin(phi);
            for j in 0..i + 1 {
                let v = dec_momenta[j];
                let v_roty = Vector4::new(
                    v.e(),
                    sin_y * v.pz() + cos_y * v.px(),
                    v.py(),
                    cos_y * v.pz() - sin_y * v.px(),
                );
                let v_rotz = Vector4::new(
                    v_roty.e(),
                    cos_z * v_roty.px() - sin_z * v_roty.py(),
                    sin_z * v_roty.px() + cos_z * v_roty.py(),
                    v_roty.pz(),
                );
                dec_momenta[j] = v_rotz;
            }
            if i == self.n - 1 {
                break;
            }

            // boost to rest frame of parent particle
            let beta = q[i] / f64::sqrt(q[i].powi(2) + int_masses[i].powi(2));
            for j in 0..i + 1 {
                let v_boosted = dec_momenta[j].boost(&Vector3::new(0.0, 0.0, -beta));
                dec_momenta[j] = v_boosted;
            }
            i += 1;
        }

        // boost to original frame of decaying particle
        for j in 0..self.n {
            let v_boosted = dec_momenta[j].boost_along(&self.decaying_p4);
            dec_momenta[j] = v_boosted;
        }
        (dec_momenta, weight)
    }
}

fn main() {
    let target = Vector4::new(0.938, 0.0, 0.0, 0.0);
    let beam = Vector4::new(0.65, 0.0, 0.0, 0.65);
    let w = beam + target;
    let masses = [0.938, 0.139, 0.139];
    let dec = Decay::new(&w, &masses);
    let mut rng = fastrand::Rng::new();
    let (p_pip_list, p_pim_list, weights): (Vec<f64>, Vec<f64>, Vec<f64>) = (0..100000)
        .map(|_| {
            let (vecs, weight) = dec.generate(&mut rng);
            let p_pip = (vecs[0] + vecs[1]).m2();
            let p_pim = (vecs[0] + vecs[2]).m2();
            (p_pip, p_pim, weight)
        })
        .multiunzip();
    // export results for plotting in Python script
    let mut map = HashMap::new();
    map.insert("m_p_pip", p_pip_list);
    map.insert("m_p_pim", p_pim_list);
    map.insert("weights", weights);
    let file_path = Path::new("rust_values.pkl");
    let file = File::create(file_path).unwrap();
    let mut writer = BufWriter::new(file);
    serde_pickle::to_writer(&mut writer, &map, Default::default()).unwrap();
}
