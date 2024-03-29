pub mod simulation {
    use rand::{thread_rng, Rng};
    use serde::{Deserialize, Serialize};
    use std::collections::VecDeque;
    use std::f64::consts::PI;
    use std::iter::FromIterator;

    #[derive(Clone, Debug)]
    pub struct Gene {
        pub gene: Vec<f64>,
    }
    pub struct GeneConst {
        pub n: f64,
        pub m: f64,
        pub theta: [f64; 8],
        pub w_on: [f64; 8],
        pub w_off: [f64; 8],
        pub w: [[f64; 8]; 8],
        pub g: [[f64; 8]; 8],
        pub w_osc: [f64; 8],
        pub w_nmj: f64,
    }
    impl Gene {
        pub fn scaling(&self) -> GeneConst {
            fn range(gene: &f64, min: f64, max: f64) -> f64 {
                (gene + 1.0) / 2.0 * (max - min) + min
            }
            //介在ニューロンと運動ニューロンの閾値 [-15.15]
            let mut theta: [f64; 8] = [0.0; 8];
            theta[0] = range(&self.gene[2], -15.0, 15.0);
            theta[1] = range(&self.gene[3], -15.0, 15.0);
            theta[2] = range(&self.gene[4], -15.0, 15.0);
            theta[3] = range(&self.gene[5], -15.0, 15.0);
            theta[4] = range(&self.gene[6], -15.0, 15.0);
            theta[5] = theta[4];
            theta[6] = range(&self.gene[7], -15.0, 15.0);
            theta[7] = theta[6];
            //感覚ニューロンONの重み [-15.15]
            let mut w_on: [f64; 8] = [0.0; 8];
            w_on[0] = range(&self.gene[8], -15.0, 15.0);
            w_on[1] = range(&self.gene[9], -15.0, 15.0);
            //感覚ニューロンOFFの重み [-15.15]
            let mut w_off: [f64; 8] = [0.0; 8];
            w_off[0] = range(&self.gene[10], -15.0, 15.0);
            w_off[1] = range(&self.gene[11], -15.0, 15.0);
            //介在ニューロンと運動ニューロンのシナプス結合の重み [-15.15]
            let mut w: [[f64; 8]; 8] = [[0.0; 8]; 8];
            w[0][2] = range(&self.gene[12], -15.0, 15.0);
            w[1][3] = range(&self.gene[13], -15.0, 15.0);
            w[2][4] = range(&self.gene[14], -15.0, 15.0);
            w[2][5] = w[2][4];
            w[3][6] = range(&self.gene[15], -15.0, 15.0);
            w[3][7] = w[3][6];
            w[4][4] = range(&self.gene[16], -15.0, 15.0);
            w[5][5] = w[4][4];
            w[6][6] = range(&self.gene[17], -15.0, 15.0);
            w[7][7] = w[6][6];
            //介在ニューロンと運動ニューロンのギャップ結合の重み [0.2.5]
            let mut g: [[f64; 8]; 8] = [[0.0; 8]; 8];
            g[0][1] = range(&self.gene[18], 0.0, 2.5);
            g[1][0] = g[0][1];
            g[2][3] = range(&self.gene[19], 0.0, 2.5);
            g[3][2] = g[2][3];
            //運動ニューロンに入る振動成分の重み [0.15]
            let mut w_osc: [f64; 8] = [0.0; 8];
            w_osc[4] = range(&self.gene[20], 0.0, 15.0);
            w_osc[7] = w_osc[4];
            w_osc[5] = -w_osc[4];
            w_osc[6] = -w_osc[4];

            GeneConst {
                n: range(&self.gene[0], 0.1, 4.2), //感覚ニューロン時間 0.1,4.2]
                m: range(&self.gene[1], 0.1, 4.2),
                theta,
                w_on,
                w_off,
                w,
                g,
                w_osc,
                w_nmj: range(&self.gene[21], 1.0, 3.0), //回転角度の重み [1,3]
            }
        }
    }
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Setting {
        pub alpha: (f64, f64),
        pub x_peak: f64,
        pub y_peak: f64,
        pub dt: f64,
        pub periodic_time: f64,
        pub frequency: f64,
        pub velocity: f64,
        pub simulation_time: f64,
        pub time_constant: f64,
    }
    pub struct Const {
        pub alpha: f64,
        pub x_peak: f64,
        pub y_peak: f64,
        pub dt: f64,
        pub periodic_time: f64,
        pub frequency: f64,
        pub velocity: f64,
        pub simulation_time: f64,
        pub time_constant: f64,
    }
    impl Setting {
        pub fn const_new(&self) -> Const {
            let mut rng = thread_rng();
            let a: f64 = rng.gen_range(self.alpha.0..self.alpha.1);
            Const {
                alpha: a,
                x_peak: self.x_peak,
                y_peak: self.y_peak,
                dt: self.dt,
                periodic_time: self.periodic_time,
                frequency: self.frequency,
                velocity: self.velocity,
                simulation_time: self.simulation_time,
                time_constant: self.time_constant,
            }
        }
    }
    #[inline]
    pub fn concentration(constant: &Const, x: f64, y: f64) -> f64 {
        constant.alpha * ((x - constant.x_peak).powi(2) + (y - constant.y_peak).powi(2)).sqrt()
    }
    #[inline]
    pub fn sigmoid(x: f64) -> f64 {
        if x > 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }
    #[allow(dead_code)]
    pub struct Time {
        pub n_time: usize,
        pub m_time: usize,
        pub simulation_time: usize,
        pub f_inv_time: usize,
        pub periodic_time: usize,
    }
    pub fn time_new(weight: &GeneConst, constant: &Const) -> Time {
        let n_time: usize = (weight.n / constant.dt).floor() as usize;
        let m_time: usize = (weight.m / constant.dt).floor() as usize;
        let simulation_time: usize = (constant.simulation_time / constant.dt).floor() as usize;
        let f_inv_time: usize = (1.0 / constant.frequency / constant.dt).floor() as usize;
        let periodic_time: usize = (constant.periodic_time / constant.dt).floor() as usize;
        Time {
            n_time,
            m_time,
            simulation_time,
            f_inv_time,
            periodic_time,
        }
    }
    #[inline]
    pub fn y_osc(constant: &Const, time: f64) -> f64 {
        (2.0 * PI * time / constant.periodic_time).sin()
    }
    #[inline]
    pub fn y_on_off(weight: &GeneConst, time: &Time, c_vec: &VecDeque<f64>) -> [f64; 2] {
        let y_on: f64 = c_vec
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                if i < time.m_time {
                    -value / weight.m
                } else {
                    value / weight.n
                }
            })
            .sum();

        if y_on < 0.0 {
            [0.0, -y_on]
        } else {
            [y_on, 0.0]
        }
    }

    pub fn chemotaxis_index(gene: &Gene, setting: &Setting) -> f64 {
        //定数
        let constant: Const = setting.const_new();
        //遺伝子の受け渡し
        let weight: GeneConst = gene.scaling();
        //時間に関する定数をステップ数に変換
        let time: Time = time_new(&weight, &constant);

        //配列の宣言
        let mut y: [[f64; 8]; 2] = [[0.0; 8]; 2];
        let mut mu: [f64; 2] = [0.0; 2];
        let mut phi: [f64; 2] = [0.0; 2];
        let mut r: [[f64; 2]; 2] = [[0.0; 2]; 2];

        //初期濃度の履歴生成
        let mut c_vec: VecDeque<f64> = VecDeque::new();
        for _ in 0..time.n_time + time.m_time {
            c_vec.push_back(concentration(&constant, 0.0, 0.0));
        }
        //運動ニューロンの初期活性を0～1の範囲でランダム化
        let _ = thread_rng().try_fill(&mut y[0][4..]);

        //ランダムな向きで配置
        let mut rng: rand::rngs::ThreadRng = thread_rng();
        mu[0] = rng.gen_range(0.0..2.0 * PI);

        let mut ci: f64 = 0.0;
        //オイラー法
        for k in 0..time.simulation_time - 1 {
            //濃度の更新
            c_vec.pop_front();
            c_vec.push_back(concentration(&constant, r[0][0], r[0][1]));

            let y_on_off: [f64; 2] = y_on_off(&weight, &time, &c_vec);
            let y_osc: f64 = y_osc(&constant, k as f64 * constant.dt);

            for i in 0..8 {
                let mut synapse: f64 = 0.0;
                let mut gap: f64 = 0.0;
                for j in 0..8 {
                    synapse += weight.w[j][i] * sigmoid(y[0][j] + weight.theta[j]);
                    gap += weight.g[j][i] * (y[0][j] - y[0][i]);
                }
                //外部からの入力
                let input: f64 = weight.w_on[i] * y_on_off[0]
                    + weight.w_off[i] * y_on_off[1]
                    + weight.w_osc[i] * y_osc;
                //ニューロンの膜電位の更新
                y[1][i] = y[0][i]
                    + (-y[0][i] + synapse + gap + input) / constant.time_constant * constant.dt;
            }

            //方向の更新
            let d: f64 = sigmoid(y[0][5] + weight.theta[5]) + sigmoid(y[0][6] + weight.theta[6]);
            let v: f64 = sigmoid(y[0][4] + weight.theta[4]) + sigmoid(y[0][7] + weight.theta[7]);
            phi[1] = phi[0];
            phi[0] = weight.w_nmj * (d - v);
            mu[1] = mu[0] + phi[0] * constant.dt;

            //位置の更新
            r[1][0] = r[0][0] + constant.velocity * (mu[0]).cos() * constant.dt;
            r[1][1] = r[0][1] + constant.velocity * (mu[0]).sin() * constant.dt;

            //走化性能指数の計算
            ci +=
                ((r[0][0] - constant.x_peak).powi(2) + (r[0][1] - constant.y_peak).powi(2)).sqrt();

            //更新
            for i in 0..8 {
                y[0][i] = y[1][i];
            }
            mu[0] = mu[1];
            for i in 0..2 {
                r[0][i] = r[1][i];
            }
        }
        //走化性能指数の計算
        ci += ((r[0][0] - constant.x_peak).powi(2) + (r[0][1] - constant.y_peak).powi(2)).sqrt();
        ci = 1.0
            - ci / (constant.x_peak.powi(2) + constant.y_peak.powi(2)).sqrt()
                / constant.simulation_time
                * constant.dt;
        if ci < 0.0 {
            ci = 0.0
        }
        ci
    }

    pub fn chemotaxis_index_wave_check(gene: &Gene, setting: &Setting) -> f64 {
        //定数
        let constant: Const = setting.const_new();
        //遺伝子の受け渡し
        let weight: GeneConst = gene.scaling();
        //時間に関する定数をステップ数に変換
        let time: Time = time_new(&weight, &constant);

        //配列の宣言
        let mut y: [[f64; 8]; 2] = [[0.0; 8]; 2];
        let mut mu: [f64; 2] = [0.0; 2];
        let mut phi: f64;
        let mut r: [[f64; 2]; 2] = [[0.0; 2]; 2];
        let mut wave_check: f64 = 0.0;
        let mut wave_point: f64 = 0.0;

        //初期濃度の履歴生成
        let mut c_vec: VecDeque<f64> = VecDeque::new();
        for _ in 0..time.n_time + time.m_time {
            c_vec.push_back(concentration(&constant, 0.0, 0.0));
        }
        //運動ニューロンの初期活性を0～1の範囲でランダム化
        let _ = thread_rng().try_fill(&mut y[0][4..]);

        //ランダムな向きで配置
        let mut rng: rand::rngs::ThreadRng = thread_rng();
        mu[0] = rng.gen_range(0.0..2.0 * PI);

        let mut ci: f64 = 0.0;
        //オイラー法
        for k in 0..time.simulation_time - 1 {
            //濃度の更新
            c_vec.pop_front();
            c_vec.push_back(concentration(&constant, r[0][0], r[0][1]));

            let y_on_off: [f64; 2] = y_on_off(&weight, &time, &c_vec);
            let y_osc: f64 = y_osc(&constant, k as f64 * constant.dt);

            for i in 0..8 {
                let mut synapse: f64 = 0.0;
                let mut gap: f64 = 0.0;
                for j in 0..8 {
                    synapse += weight.w[j][i] * sigmoid(y[0][j] + weight.theta[j]);
                    gap += weight.g[j][i] * (y[0][j] - y[0][i]);
                }
                //外部からの入力
                let input: f64 = weight.w_on[i] * y_on_off[0]
                    + weight.w_off[i] * y_on_off[1]
                    + weight.w_osc[i] * y_osc;
                //ニューロンの膜電位の更新
                y[1][i] = y[0][i]
                    + (-y[0][i] + synapse + gap + input) / constant.time_constant * constant.dt;
            }

            //方向の更新
            let d: f64 = sigmoid(y[0][5] + weight.theta[5]) + sigmoid(y[0][6] + weight.theta[6]);
            let v: f64 = sigmoid(y[0][4] + weight.theta[4]) + sigmoid(y[0][7] + weight.theta[7]);
            phi = weight.w_nmj * (d - v);
            mu[1] = mu[0] + phi * constant.dt;

            //ピルエット
            if k % time.f_inv_time == time.f_inv_time - 1 {
                let mut rng: rand::rngs::ThreadRng = thread_rng();
                mu[1] = rng.gen_range(0.0..2.0 * PI);
            }

            //波のチェック
            if k % time.periodic_time == time.periodic_time / 4 {
                wave_check = phi;
            } else if k % time.periodic_time == time.periodic_time * 3 / 4 && wave_check * phi > 0.0
            {
                wave_point += 0.008;
            }

            //位置の更新
            r[1][0] = r[0][0] + constant.velocity * (mu[0]).cos() * constant.dt;
            r[1][1] = r[0][1] + constant.velocity * (mu[0]).sin() * constant.dt;

            //走化性能指数の計算
            ci +=
                ((r[0][0] - constant.x_peak).powi(2) + (r[0][1] - constant.y_peak).powi(2)).sqrt();

            //更新
            for i in 0..8 {
                y[0][i] = y[1][i];
            }
            mu[0] = mu[1];
            for i in 0..2 {
                r[0][i] = r[1][i];
            }
        }
        //走化性能指数の計算
        ci += ((r[0][0] - constant.x_peak).powi(2) + (r[0][1] - constant.y_peak).powi(2)).sqrt();
        ci = 1.0
            - ci / (constant.x_peak.powi(2) + constant.y_peak.powi(2)).sqrt()
                / constant.simulation_time
                * constant.dt
            - wave_point;
        if ci < 0.0 {
            ci = 0.0
        }
        ci
    }

    impl FromIterator<f64> for Gene {
        fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
            let gene: Vec<f64> = iter.into_iter().collect();
            Gene { gene }
        }
    }
}

pub mod genetic_algorithm {
    use crate::simulation::*;
    use rand::{thread_rng, Rng};
    use rand_distr::{Distribution, Normal};
    use rayon::prelude::*;
    use serde::{Deserialize, Serialize};
    use std::fs::File;
    use std::io::prelude::*;
    use toml::Value;

    #[derive(Clone, Debug)]
    pub struct Ga {
        pub value: f64,
        pub gene: Gene,
    }

    impl Ga {
        pub fn fitness(&self, setting: &Setting, average: i32, version: i32) -> f64 {
            let mut sum: f64 = 0.0;
            if version == 0 {
                for _ in 0..average {
                    sum += chemotaxis_index(&self.gene, setting)
                }
            } else if version == 1 {
                for _ in 0..average {
                    sum += chemotaxis_index_wave_check(&self.gene, setting)
                }
            }
            sum / average as f64
        }
    }

    pub fn population_new(gen_size: usize, pop_size: usize) -> Vec<Ga> {
        let mut rng: rand::rngs::ThreadRng = thread_rng();
        let population: Vec<Ga> = (0..pop_size)
            .map(|_| {
                let gene: Gene = (0..gen_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect::<Gene>();
                Ga { value: 0.0, gene }
            })
            .collect();
        population
    }

    pub fn evaluate_fitness(
        population: &mut Vec<Ga>,
        setting: &Setting,
        average: i32,
        version: i32,
    ) -> Vec<Ga> {
        population
            .par_iter_mut()
            .map(|ind| Ga {
                value: ind.fitness(setting, average, version),
                gene: ind.gene.clone(),
            })
            .collect()
    }

    pub fn two_point_crossover(parent1: &Ga, parent2: &Ga) -> Vec<Ga> {
        let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
        let crossover_points: (usize, usize) = (
            rng.gen_range(0..parent1.gene.gene.len()),
            rng.gen_range(0..parent1.gene.gene.len()),
        );
        let (start, end) = if crossover_points.0 < crossover_points.1 {
            (crossover_points.0, crossover_points.1)
        } else {
            (crossover_points.1, crossover_points.0)
        };

        let mut child_gene1: Gene = parent1.gene.clone();
        let mut child_gene2: Gene = parent2.gene.clone();
        for i in start..end {
            child_gene1.gene[i] = parent2.gene.gene[i];
            child_gene2.gene[i] = parent1.gene.gene[i];
        }

        let child1: Ga = Ga {
            value: 0.0,
            gene: child_gene1,
        };
        let child2: Ga = Ga {
            value: 0.0,
            gene: child_gene2,
        };
        vec![child1, child2]
    }

    pub fn mutation(gene: &Ga, rate: f64, mean_std: (f64, f64)) -> Ga {
        let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
        let normal =
            Normal::new(mean_std.0, mean_std.1).expect("Failed to create normal distribution");

        let mut mutated_gene: Gene = gene.gene.clone();

        for val in mutated_gene.gene.iter_mut() {
            if rng.gen::<f64>() < rate {
                let delta: f64 = normal.sample(&mut rng);
                *val += delta;
                if *val > 1.0 {
                    *val = 1.0
                } else if *val < -1.0 {
                    *val = -1.0
                }
            }
        }

        Ga {
            value: 0.0,
            gene: mutated_gene,
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Gajson {
        pub value: f64,
        pub gene: Vec<f64>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Gasetting {
        pub average: i32,
        pub gen_size: usize,
        pub ga_count: usize,
        pub n_gen: usize,
        pub pop_size: usize,
        pub sel_top: usize,
        pub mat_pb: f64,
        pub mut_pb: f64,
        pub re_val: usize,
    }

    pub fn genetic_algorithm() {
        // GA_setting.toml ファイルを読み込む
        let toml_str: String =
            std::fs::read_to_string("GA_setting.toml").expect("Failed to read file");
        let value: Value = toml::from_str(&toml_str).expect("Failed to parse TOML");

        // 各セクションを取り出す
        let ga_setting: Gasetting = value["Ga_setting"]
            .clone()
            .try_into()
            .expect("Failed to parse Gasetting");
        let setting: Setting = value["setting"]
            .clone()
            .try_into()
            .expect("Failed to parse Setting");
        let testing: Setting = value["testing"]
            .clone()
            .try_into()
            .expect("Failed to parse Testing");

        // 途中結果の書き出し用のtxtファイルの作成
        let mut file = File::create("Result.txt").unwrap();

        //遺伝的アルゴリズムの結果
        let mut result: Vec<Ga> = Vec::new();

        for count in 0..ga_setting.ga_count {
            //初期集団を生成
            let mut population: Vec<Ga> = population_new(ga_setting.gen_size, ga_setting.pop_size);

            //個体の評価(version:0は通常、version:1は波打つかチェックしている)
            let mut evaluate: Vec<Ga> =
                evaluate_fitness(&mut population, &setting, ga_setting.average, 1);

            //個体をvalueの値によって降順でsort
            evaluate.sort_by(|a: &Ga, b: &Ga| b.value.partial_cmp(&a.value).unwrap());

            println!(
                "{:3}_Gen: {:03}, Fitness_1: {:.5}",
                count + 1,
                0,
                evaluate[0].value
            );
            println!("              Fitness_2: {:.5}", evaluate[1].value);
            println!("              Fitness_3: {:.5}", evaluate[2].value);
            println!();

            for i in 0..ga_setting.n_gen {
                //選択
                let select: Vec<Ga> = evaluate.iter().take(ga_setting.sel_top).cloned().collect();

                //交叉
                let mut mate: Vec<Ga> = Vec::new();
                let clone: Vec<Ga> = select.clone();
                let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
                for i in (0..clone.len()).step_by(2) {
                    if i + 1 < clone.len() && rng.gen::<f64>() < ga_setting.mat_pb {
                        let parent1: &Ga = &clone[i];
                        let parent2: &Ga = &clone[i + 1];
                        let child: Vec<Ga> = two_point_crossover(parent1, parent2);
                        mate.extend(child);
                    }
                }

                //変異
                let mut mutant: Vec<Ga> = Vec::new();
                let clone: Vec<Ga> = select.clone();
                let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
                for ind in clone.iter() {
                    if rng.gen::<f64>() < ga_setting.mut_pb {
                        mutant.push(mutation(ind, 0.4, (0.0, 0.05)));
                    }
                }

                if i % ga_setting.re_val == 0 {
                    //再評価
                    let mut offspring: Vec<Ga> = Vec::new();
                    offspring.extend(select);
                    offspring.extend(mate);
                    offspring.extend(mutant);
                    let population: Vec<Ga> =
                        population_new(ga_setting.gen_size, ga_setting.pop_size - offspring.len());
                    offspring.extend(population);
                    evaluate.clear();
                    let offspring_evaluate: Vec<Ga> =
                        evaluate_fitness(&mut offspring, &setting, ga_setting.average, 1);
                    evaluate.extend(offspring_evaluate);
                    evaluate.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
                } else {
                    //通常評価
                    let mut offspring: Vec<Ga> = Vec::new();
                    offspring.extend(mate);
                    offspring.extend(mutant);
                    let population: Vec<Ga> = population_new(
                        ga_setting.gen_size,
                        ga_setting.pop_size - ga_setting.sel_top - offspring.len(),
                    );
                    offspring.extend(population);
                    evaluate.clear();
                    let offspring_evaluate: Vec<Ga> =
                        evaluate_fitness(&mut offspring, &setting, ga_setting.average, 1);
                    evaluate.extend(offspring_evaluate);
                    evaluate.extend(select);
                    evaluate.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
                }

                println!(
                    "{:3}_Gen: {:03}, Fitness_1: {:.5}",
                    count + 1,
                    i + 1,
                    evaluate[0].value
                );
                println!("              Fitness_2: {:.5}", evaluate[1].value);
                println!("              Fitness_3: {:.5}", evaluate[2].value);
                println!();
            }

            //最も優秀な個体を結果に格納
            result.push(evaluate[0].clone());

            // 途中結果の書き出し
            let result_gajson: Gajson = Gajson {
                value: evaluate[0].value,
                gene: evaluate[0].gene.gene.clone(),
            };
            let json_string = serde_json::to_string_pretty(&result_gajson).unwrap();
            file.write_all(json_string.as_bytes()).unwrap();
            writeln!(file, ",").unwrap();
        }

        //正しいCIを用いて結果を評価する
        let mut result_evaluate: Vec<Ga> =
            evaluate_fitness(&mut result, &testing, ga_setting.average, 0);
        result_evaluate.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        //Gajsonに変換
        let result_evaluate_gajson: Vec<Gajson> = result_evaluate
            .into_iter()
            .map(|ga: Ga| Gajson {
                value: ga.value,
                gene: ga.gene.gene,
            })
            .collect();

        //JSON文字列にシリアライズ
        let result_json = serde_json::to_string_pretty(&result_evaluate_gajson).unwrap();

        //JSON文字列をファイルに書き込む
        let mut file: File = File::create("Result.json").expect("ファイルの作成に失敗しました");
        file.write_all(result_json.as_bytes())
            .expect("ファイルへの書き込みに失敗しました");
    }

    #[allow(unused_variables)]
    pub fn population_new_optimize_aser_aiy_positive(gen_size: usize, pop_size: usize) -> Vec<Ga> {
        let mut rng: rand::rngs::ThreadRng = thread_rng();
        let population: Vec<Ga> = (0..pop_size)
            .map(|_| {
                let gene: Gene = Gene {
                    gene: vec![
                        -0.8094022576319283,
                        -0.6771492613425638,
                        0.05892807075993428,
                        -0.4894407617977082,
                        0.1593721867510597,
                        0.3576592038271041,
                        -0.5664294232926526,
                        -0.7853343958692636,
                        0.6552003805912084,
                        -0.6492992485125678,
                        rng.gen_range(0.0..1.0),
                        rng.gen_range(0.0..1.0),
                        -1.0,
                        -0.7386107983898611,
                        0.02074396537515929,
                        0.7150315462816783,
                        -0.9243504880454858,
                        0.1353396882729762,
                        0.9494528443702027,
                        0.7727883271643218,
                        -0.6046043758402895,
                        0.7969062294208619,
                    ],
                };

                Ga { value: 0.0, gene }
            })
            .collect();
        population
    }

    pub fn mutation_aiy_optimize_aser_aiy_positive(
        gene: &Ga,
        rate: f64,
        mean_std: (f64, f64),
    ) -> Ga {
        let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
        let normal: Normal<f64> =
            Normal::new(mean_std.0, mean_std.1).expect("Failed to create normal distribution");

        let mut mutated_gene: Gene = gene.gene.clone();

        for (index, val) in mutated_gene.gene.iter_mut().enumerate() {
            if index > 9 && index < 12 {
                // 遺伝子番号10〜11(w_offの結合)
                if rng.gen::<f64>() < rate {
                    let delta: f64 = normal.sample(&mut rng);
                    *val += delta;
                    if *val < 0.0 {
                        *val -= delta
                    } else if *val > 1.0 {
                        *val = 1.0
                    }
                }
            }
        }

        Ga {
            value: 0.0,
            gene: mutated_gene,
        }
    }

    pub fn population_new_aiz_negative(gen_size: usize, pop_size: usize) -> Vec<Ga> {
        let mut rng: rand::rngs::ThreadRng = thread_rng();
        let population: Vec<Ga> = (0..pop_size)
            .map(|_| {
                let mut gene: Gene = (0..gen_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect::<Gene>();

                // 遺伝子番号12〜13(w_02, w_13の結合)
                for i in 12..14 {
                    if gene.gene[i] > 0.0 {
                        gene.gene[i] = -gene.gene[i]
                    }
                }

                Ga { value: 0.0, gene }
            })
            .collect();
        population
    }

    pub fn mutation_aiz_negative(gene: &Ga, rate: f64, mean_std: (f64, f64)) -> Ga {
        let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
        let normal: Normal<f64> =
            Normal::new(mean_std.0, mean_std.1).expect("Failed to create normal distribution");

        let mut mutated_gene: Gene = gene.gene.clone();

        for (index, val) in mutated_gene.gene.iter_mut().enumerate() {
            if index > 11 && index < 14 {
                // 遺伝子番号12〜13(w_02, w_13の結合)
                if rng.gen::<f64>() < rate {
                    let delta: f64 = normal.sample(&mut rng);
                    *val += delta;
                    if *val > 0.0 {
                        *val -= delta
                    } else if *val < -1.0 {
                        *val = -1.0
                    }
                }
            } else if rng.gen::<f64>() < rate {
                let delta: f64 = normal.sample(&mut rng);
                *val += delta;
                if *val > 1.0 {
                    *val = 1.0;
                } else if *val < -1.0 {
                    *val = -1.0;
                }
            }
        }

        Ga {
            value: 0.0,
            gene: mutated_gene,
        }
    }

    pub fn genetic_algorithm_biologically_correct() {
        // GA_setting.toml ファイルを読み込む
        let toml_str: String =
            std::fs::read_to_string("GA_setting.toml").expect("Failed to read file");
        let value: Value = toml::from_str(&toml_str).expect("Failed to parse TOML");

        // 各セクションを取り出す
        let ga_setting: Gasetting = value["Ga_setting"]
            .clone()
            .try_into()
            .expect("Failed to parse Gasetting");
        let setting: Setting = value["setting"]
            .clone()
            .try_into()
            .expect("Failed to parse Setting");
        let testing: Setting = value["testing"]
            .clone()
            .try_into()
            .expect("Failed to parse Testing");

        // 途中結果の書き出し用のtxtファイルの作成
        let mut file = File::create("Result.txt").unwrap();

        //遺伝的アルゴリズムの結果
        let mut result: Vec<Ga> = Vec::new();

        for count in 0..ga_setting.ga_count {
            //初期集団を生成
            let mut population: Vec<Ga> =
                population_new_optimize_aser_aiy_positive(ga_setting.gen_size, ga_setting.pop_size);

            //個体の評価(version:0は通常、version:1は波打つかチェックしている)
            let mut evaluate: Vec<Ga> =
                evaluate_fitness(&mut population, &setting, ga_setting.average, 1);

            //個体をvalueの値によって降順でsort
            evaluate.sort_by(|a: &Ga, b: &Ga| b.value.partial_cmp(&a.value).unwrap());

            println!(
                "{:3}_Gen: {:03}, Fitness_1: {:.5}",
                count + 1,
                0,
                evaluate[0].value
            );
            println!("              Fitness_2: {:.5}", evaluate[1].value);
            println!("              Fitness_3: {:.5}", evaluate[2].value);
            println!();

            for i in 0..ga_setting.n_gen {
                //選択
                let select: Vec<Ga> = evaluate.iter().take(ga_setting.sel_top).cloned().collect();

                //交叉
                let mut mate: Vec<Ga> = Vec::new();
                let clone: Vec<Ga> = select.clone();
                let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
                for i in (0..clone.len()).step_by(2) {
                    if i + 1 < clone.len() && rng.gen::<f64>() < ga_setting.mat_pb {
                        let parent1: &Ga = &clone[i];
                        let parent2: &Ga = &clone[i + 1];
                        let child: Vec<Ga> = two_point_crossover(parent1, parent2);
                        mate.extend(child);
                    }
                }

                //変異
                let mut mutant: Vec<Ga> = Vec::new();
                let clone: Vec<Ga> = select.clone();
                let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
                for ind in clone.iter() {
                    if rng.gen::<f64>() < ga_setting.mut_pb {
                        mutant.push(mutation_aiy_optimize_aser_aiy_positive(
                            ind,
                            0.4,
                            (0.0, 0.05),
                        ));
                    }
                }

                if i % ga_setting.re_val == 0 {
                    //再評価
                    let mut offspring: Vec<Ga> = Vec::new();
                    offspring.extend(select);
                    offspring.extend(mate);
                    offspring.extend(mutant);
                    let population: Vec<Ga> = population_new_optimize_aser_aiy_positive(
                        ga_setting.gen_size,
                        ga_setting.pop_size - offspring.len(),
                    );
                    offspring.extend(population);
                    evaluate.clear();
                    let offspring_evaluate: Vec<Ga> =
                        evaluate_fitness(&mut offspring, &setting, ga_setting.average, 1);
                    evaluate.extend(offspring_evaluate);
                    evaluate.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
                } else {
                    //通常評価
                    let mut offspring: Vec<Ga> = Vec::new();
                    offspring.extend(mate);
                    offspring.extend(mutant);
                    let population: Vec<Ga> = population_new_optimize_aser_aiy_positive(
                        ga_setting.gen_size,
                        ga_setting.pop_size - ga_setting.sel_top - offspring.len(),
                    );
                    offspring.extend(population);
                    evaluate.clear();
                    let offspring_evaluate: Vec<Ga> =
                        evaluate_fitness(&mut offspring, &setting, ga_setting.average, 1);
                    evaluate.extend(offspring_evaluate);
                    evaluate.extend(select);
                    evaluate.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());
                }

                println!(
                    "{:3}_Gen: {:03}, Fitness_1: {:.5}",
                    count + 1,
                    i + 1,
                    evaluate[0].value
                );
                println!("              Fitness_2: {:.5}", evaluate[1].value);
                println!("              Fitness_3: {:.5}", evaluate[2].value);
                println!();
            }

            //最も優秀な個体を結果に格納
            result.push(evaluate[0].clone());

            // 途中結果の書き出し
            let result_gajson: Gajson = Gajson {
                value: evaluate[0].value,
                gene: evaluate[0].gene.gene.clone(),
            };
            let json_string = serde_json::to_string_pretty(&result_gajson).unwrap();
            file.write_all(json_string.as_bytes()).unwrap();
            writeln!(file, ",").unwrap();
        }

        //正しいCIを用いて結果を評価する
        let mut result_evaluate: Vec<Ga> =
            evaluate_fitness(&mut result, &testing, ga_setting.average, 0);
        result_evaluate.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        //Gajsonに変換
        let result_evaluate_gajson: Vec<Gajson> = result_evaluate
            .into_iter()
            .map(|ga: Ga| Gajson {
                value: ga.value,
                gene: ga.gene.gene,
            })
            .collect();

        //JSON文字列にシリアライズ
        let result_json: String = serde_json::to_string_pretty(&result_evaluate_gajson).unwrap();

        //JSON文字列をファイルに書き込む
        let mut file: File = File::create("Result_aiz_negative_optimize_aser_aiy_positive.json")
            .expect("ファイルの作成に失敗しました");
        file.write_all(result_json.as_bytes())
            .expect("ファイルへの書き込みに失敗しました");
    }
}

pub mod analysis {
    use crate::genetic_algorithm::*;
    use crate::simulation::*;
    use rand::{thread_rng, Rng};
    use rayon::prelude::*;
    use serde::{Deserialize, Serialize};
    use std::collections::VecDeque;
    use std::f64::consts::PI;
    use std::fs::File;
    use std::io::Write;
    use toml::Value;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Filename {
        pub read_result_json: String,
        pub bearing_vs_turning_bias_output: String,
        pub nomal_gradient_vs_turning_bias_output: String,
        pub translational_gradient_vs_turning_bias_output: String,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Analysisusefunction {
        pub mode: Vec<i32>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Analysisusegene {
        pub mode: usize,
        pub gene_numbers: Vec<usize>,
        pub gene_number_range: Vec<usize>,
    }

    pub fn read_result() -> Vec<Ga> {
        // klinotaxis_analysis.toml ファイルを読み込む
        let toml_str: String =
            std::fs::read_to_string("klinotaxis_analysis.toml").expect("Failed to read file");
        let value: Value = toml::from_str(&toml_str).expect("Failed to parse TOML");
        let file_name: Filename = value["file_name"]
            .clone()
            .try_into()
            .expect("Failed to parse Setting");

        let result_json: String = std::fs::read_to_string(file_name.read_result_json)
            .expect("ファイルの読み込みに失敗しました");

        // JSON 文字列を Vec<Gajson> にデシリアライズ
        let result_gajson: Vec<Gajson> =
            serde_json::from_str(&result_json).expect("JSONのデシリアライズに失敗しました");

        // Gajson を Ga に変換
        let result: Vec<Ga> = result_gajson
            .into_iter()
            .map(|gajson: Gajson| Ga {
                value: gajson.value,
                gene: Gene { gene: gajson.gene },
            })
            .collect();
        result
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Gausssetting {
        pub c_0: f64,
        pub lambda: f64,
    }

    #[inline]
    pub fn gauss_concentration(
        gauss_setting: &Gausssetting,
        constant: &Const,
        x: f64,
        y: f64,
    ) -> f64 {
        gauss_setting.c_0
            * (-((x - constant.x_peak).powi(2) + (y - constant.y_peak).powi(2))
                / (2.0 * gauss_setting.lambda.powi(2)))
            .exp()
    }

    pub fn two_gauss_concentration(
        gauss_setting: &Gausssetting,
        constant: &Const,
        x: f64,
        y: f64,
    ) -> f64 {
        gauss_setting.c_0
            * ((-((x - constant.x_peak).powi(2) + (y - constant.y_peak).powi(2))
                / (2.0 * gauss_setting.lambda.powi(2)))
            .exp()
                - (-((x + constant.x_peak).powi(2) + (y + constant.y_peak).powi(2))
                    / (2.0 * gauss_setting.lambda.powi(2)))
                .exp())
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Analysissetting {
        pub gene_number: usize,
        pub mode: usize,
        pub analysis_loop: usize,
        pub periodic_number: usize,
        pub periodic_number_drain: usize,
        pub bin_range: usize,
        pub delta: f64,
        pub bin_number: usize,
        pub concentration_gradient_max: f64,
    }

    #[allow(clippy::too_many_arguments)]
    pub fn klinotaxis_analysis(
        gene: &Gene,
        setting: &Setting,
        gauss_setting: &Gausssetting,
        mode: usize,
        periodic_number: usize,
        periodic_number_drain: usize,
        delta: f64,
        analysis_use_function: &i32,
    ) -> Vec<Vec<f64>> {
        //定数
        let constant: Const = setting.const_new();
        //遺伝子の受け渡し
        let weight: GeneConst = gene.scaling();
        //時間に関する定数をステップ数に変換
        let time: Time = time_new(&weight, &constant);

        //配列の宣言
        let mut y: [[f64; 8]; 2] = [[0.0; 8]; 2];
        let mut mu: [f64; 2] = [0.0; 2];
        let mut phi: [f64; 2] = [0.0; 2];
        let mut r: [[f64; 2]; 2] = [[0.0; 2]; 2];

        //Vecの宣言
        let mut r_vec: Vec<[f64; 2]> = vec![[0.0; 2]; 1];
        let mut mu_vec: Vec<f64> = vec![0.0];

        //初期濃度の履歴生成
        let mut c_vec: VecDeque<f64> = VecDeque::new();
        if mode == 0 {
            for _ in 0..time.n_time + time.m_time {
                c_vec.push_back(concentration(&constant, 0.0, 0.0));
            }
        } else if mode == 1 {
            for _ in 0..time.n_time + time.m_time {
                c_vec.push_back(gauss_concentration(gauss_setting, &constant, 0.0, 0.0));
            }
        } else if mode == 2 {
            for _ in 0..time.n_time + time.m_time {
                c_vec.push_back(two_gauss_concentration(gauss_setting, &constant, 0.0, 0.0));
            }
        }

        //運動ニューロンの初期活性を0～1の範囲でランダム化
        let _ = thread_rng().try_fill(&mut y[0][4..]);

        //ランダムな向きで配置
        let mut rng: rand::rngs::ThreadRng = thread_rng();
        mu[0] = rng.gen_range(0.0..2.0 * PI);

        //オイラー法
        for k in 0..time.simulation_time - 1 {
            //濃度の更新
            c_vec.pop_front();
            if mode == 0 {
                c_vec.push_back(concentration(&constant, r[0][0], r[0][1]));
            } else if mode == 1 {
                c_vec.push_back(gauss_concentration(
                    gauss_setting,
                    &constant,
                    r[0][0],
                    r[0][1],
                ));
            } else if mode == 2 {
                c_vec.push_back(two_gauss_concentration(
                    gauss_setting,
                    &constant,
                    r[0][0],
                    r[0][1],
                ));
            }

            let y_on_off: [f64; 2] = y_on_off(&weight, &time, &c_vec);
            let y_osc: f64 = y_osc(&constant, k as f64 * constant.dt);

            for i in 0..8 {
                let mut synapse: f64 = 0.0;
                let mut gap: f64 = 0.0;
                for j in 0..8 {
                    synapse += weight.w[j][i] * sigmoid(y[0][j] + weight.theta[j]);
                    gap += weight.g[j][i] * (y[0][j] - y[0][i]);
                }
                //外部からの入力
                let input: f64 = weight.w_on[i] * y_on_off[0]
                    + weight.w_off[i] * y_on_off[1]
                    + weight.w_osc[i] * y_osc;
                //ニューロンの膜電位の更新
                y[1][i] = y[0][i]
                    + (-y[0][i] + synapse + gap + input) / constant.time_constant * constant.dt;
            }

            //方向の更新
            let d: f64 = sigmoid(y[0][5] + weight.theta[5]) + sigmoid(y[0][6] + weight.theta[6]);
            let v: f64 = sigmoid(y[0][4] + weight.theta[4]) + sigmoid(y[0][7] + weight.theta[7]);
            phi[1] = phi[0];
            phi[0] = weight.w_nmj * (d - v);
            mu[1] = mu[0] + phi[0] * constant.dt;

            //位置の更新
            r[1][0] = r[0][0] + constant.velocity * (mu[0]).cos() * constant.dt;
            r[1][1] = r[0][1] + constant.velocity * (mu[0]).sin() * constant.dt;

            //Vecへの追加
            r_vec.push(r[1]);
            mu_vec.push(mu[1]);

            //更新
            for i in 0..8 {
                y[0][i] = y[1][i];
            }
            mu[0] = mu[1];
            for i in 0..2 {
                r[0][i] = r[1][i];
            }
        }

        let mut x_axis_value: Vec<f64> = Vec::new();
        if *analysis_use_function == 0 {
            // Bearing
            x_axis_value = bearing(&r_vec, &constant, &time, &periodic_number);
        } else if *analysis_use_function == 1 {
            // Normal gradient
            x_axis_value = normal_gradient(
                &r_vec,
                &constant,
                gauss_setting,
                &time,
                &periodic_number,
                delta,
                mode,
            );
        } else if *analysis_use_function == 2 {
            // Translational gradient
            x_axis_value = translational_gradient(
                &r_vec,
                &constant,
                gauss_setting,
                &time,
                &periodic_number,
                delta,
                mode,
            );
        }

        // Turning bias
        let mut y_axis_value: Vec<f64> = turning_bias_bear(&r_vec, &time, &periodic_number);

        // 先頭の要素を削除
        x_axis_value.drain(..periodic_number_drain * time.periodic_time);
        y_axis_value.drain(..periodic_number_drain * time.periodic_time);

        // 結果
        let result: Vec<Vec<f64>> = vec![x_axis_value, y_axis_value];

        result
    }

    pub fn bearing(
        r_vec: &[[f64; 2]],
        constant: &Const,
        time: &Time,
        periodic_number: &usize,
    ) -> Vec<f64> {
        let mut bearing_point: Vec<[[f64; 2]; 2]> = Vec::new();
        for i in 0..time.simulation_time - 2 * periodic_number * time.periodic_time {
            bearing_point.push([r_vec[i], r_vec[i + periodic_number * time.periodic_time]]);
        }

        let peak_vec: [f64; 2] = [constant.x_peak, constant.y_peak];
        let mut bearing: Vec<f64> = Vec::new();

        for bearing_point_item in &bearing_point {
            let bearing_vec: [f64; 2] = [
                bearing_point_item[1][0] - bearing_point_item[0][0],
                bearing_point_item[1][1] - bearing_point_item[0][1],
            ];
            let dot_product: f64 = peak_vec[0] * bearing_vec[0] + peak_vec[1] * bearing_vec[1];
            let magnitude_peak: f64 = (peak_vec[0].powi(2) + peak_vec[1].powi(2)).sqrt();
            let magnitude_bearing: f64 = (bearing_vec[0].powi(2) + bearing_vec[1].powi(2)).sqrt();
            let angle_radian: f64 = (dot_product / (magnitude_peak * magnitude_bearing)).acos();
            let mut angle_degrees: f64 = angle_radian.to_degrees();
            let cross_product: f64 = peak_vec[0] * bearing_vec[1] - peak_vec[1] * bearing_vec[0];
            if cross_product < 0.0 {
                angle_degrees *= -1.0;
            }
            bearing.push(angle_degrees);
        }

        bearing
    }

    pub fn turning_bias_bear(r_vec: &[[f64; 2]], time: &Time, periodic_number: &usize) -> Vec<f64> {
        //  ターニングバイアス計算用の座標リストを初期化
        let mut turning_bias_point: Vec<[[f64; 2]; 2]> = Vec::new();
        //  指定された時間範囲内でターニングバイアスの座標リストを生成
        for i in 0..time.simulation_time - periodic_number * time.periodic_time {
            turning_bias_point.push([r_vec[i], r_vec[i + periodic_number * time.periodic_time]]);
        }
        //  ターニングバイアス計算用のベクトルリストを初期化
        let mut turning_bias_vec: Vec<[[f64; 2]; 2]> = Vec::new();
        //  ターニングバイアス計算用の距離リストを初期化
        let mut turning_bias_distance: Vec<f64> = Vec::new();

        //  指定された時間範囲内でターニングバイアスのベクトル・距離リストを生成
        for i in 0..time.simulation_time - 2 * periodic_number * time.periodic_time {
            let turning_bias_vec_1: [f64; 2] = [
                turning_bias_point[i][1][0] - turning_bias_point[i][0][0],
                turning_bias_point[i][1][1] - turning_bias_point[i][0][1],
            ];
            let turning_bias_vec_2: [f64; 2] = [
                turning_bias_point[i + periodic_number * time.periodic_time][1][0]
                    - turning_bias_point[i + periodic_number * time.periodic_time][0][0],
                turning_bias_point[i + periodic_number * time.periodic_time][1][1]
                    - turning_bias_point[i + periodic_number * time.periodic_time][0][1],
            ];
            turning_bias_vec.push([turning_bias_vec_1, turning_bias_vec_2]);

            let distance: f64 = (turning_bias_vec_1[0].powi(2) + turning_bias_vec_1[1].powi(2))
                .sqrt()
                + (turning_bias_vec_2[0].powi(2) + turning_bias_vec_2[1].powi(2)).sqrt();
            turning_bias_distance.push(distance);
        }

        //  ターニングバイアスの計算用の角度リストを初期化
        let mut turning_bias_angle: Vec<f64> = Vec::new();

        //  ターニングバイアスのベクトルから角度を計算
        for turing_bias_vec_item in &turning_bias_vec {
            let dot_product: f64 = turing_bias_vec_item[0][0] * turing_bias_vec_item[1][0]
                + turing_bias_vec_item[0][1] * turing_bias_vec_item[1][1];
            let magnitude_vec_1: f64 =
                (turing_bias_vec_item[0][0].powi(2) + turing_bias_vec_item[0][1].powi(2)).sqrt();
            let magnitude_vec_2: f64 =
                (turing_bias_vec_item[1][0].powi(2) + turing_bias_vec_item[1][1].powi(2)).sqrt();
            let angle_radian: f64 = (dot_product / (magnitude_vec_1 * magnitude_vec_2)).acos();
            let mut angle_degrees: f64 = angle_radian.to_degrees();
            let cross_product: f64 = turing_bias_vec_item[0][0] * turing_bias_vec_item[1][1]
                - turing_bias_vec_item[0][1] * turing_bias_vec_item[1][0];
            if cross_product < 0.0 {
                angle_degrees *= -1.0;
            }
            turning_bias_angle.push(angle_degrees);
        }

        //  ターニングバイアスの計算結果用のリストを初期化
        let mut turning_bias: Vec<f64> = Vec::new();

        //  ターニングバイアスの計算
        for (angle, distance) in turning_bias_angle.iter().zip(&turning_bias_distance) {
            let bias = angle / distance;
            turning_bias.push(bias);
        }

        turning_bias
    }

    pub fn histgram_count_bearing(
        bearing: &[f64],
        turning_bias: &[f64],
        bin_range: usize,
    ) -> Vec<f64> {
        //ヒストグラムの種
        let mut turning_bias_mean: Vec<f64> = Vec::new();

        for i in (-180..180).step_by(bin_range) {
            let mut mean: Vec<f64> = Vec::new();
            for (j, &bearing_value) in bearing.iter().enumerate() {
                if ((i as f64) < bearing_value) && (bearing_value < (i as f64 + bin_range as f64)) {
                    mean.push(turning_bias[j]);
                }
            }
            // 平均値の計算
            let mean_value: f64 = if !mean.is_empty() {
                mean.iter().sum::<f64>() / mean.len() as f64
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };
            turning_bias_mean.push(mean_value);
        }

        turning_bias_mean
    }

    pub fn analysis_klinotaxis_bearing_errbar_std_max_min(
        result_ga: &[Ga],
        file_name: &Filename,
        liner_setting: &Setting,
        gauss_setting: &Gausssetting,
        analysis_setting: &Analysissetting,
        analysis_use_function: &i32,
    ) {
        let hist_mean: Vec<Vec<f64>> = (0..analysis_setting.analysis_loop)
            .into_par_iter()
            .map(|_| {
                let result: Vec<Vec<f64>> = klinotaxis_analysis(
                    &result_ga[analysis_setting.gene_number].gene,
                    liner_setting,
                    gauss_setting,
                    analysis_setting.mode,
                    analysis_setting.periodic_number,
                    analysis_setting.periodic_number_drain,
                    analysis_setting.delta,
                    analysis_use_function,
                );

                histgram_count_bearing(&result[0], &result[1], analysis_setting.bin_range)
            })
            .collect::<Vec<Vec<f64>>>();

        // ヒストグラムの作成
        let mut bearing_hist: Vec<f64> = Vec::new();
        let mut turning_bias_hist: Vec<f64> = Vec::new();
        let mut error_bar_std: Vec<f64> = Vec::new();
        let mut error_bar_max: Vec<f64> = Vec::new();
        let mut error_bar_min: Vec<f64> = Vec::new();

        for (count, i) in (-180..180).step_by(analysis_setting.bin_range).enumerate() {
            let mut mean: Vec<f64> = Vec::new();
            for row in &hist_mean {
                if !row[count].is_nan() {
                    mean.push(row[count]);
                }
            }

            // 平均値の計算
            let mean_value: f64 = if !mean.is_empty() {
                mean.iter().sum::<f64>() / mean.len() as f64
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };

            // エラーバーの計算
            let max: f64 = if !mean.is_empty() {
                mean.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            } else {
                f64::NAN
            };

            let min: f64 = if !mean.is_empty() {
                mean.iter().cloned().fold(f64::INFINITY, f64::min)
            } else {
                f64::NAN
            };

            // 標準偏差の計算
            let std: f64 = if !mean.is_empty() {
                let variance: f64 =
                    mean.iter().map(|&x| (x - mean_value).powi(2)).sum::<f64>() / mean.len() as f64;
                variance.sqrt()
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };

            bearing_hist.push(i as f64);
            turning_bias_hist.push(mean_value);
            error_bar_std.push(std);
            error_bar_max.push(max);
            error_bar_min.push(min);
        }

        // Open a file for writing
        let mut file: File = File::create(&file_name.bearing_vs_turning_bias_output).unwrap();

        // Iterate over the vectors and write each triplet of values to a line in the file
        for ((((bearing, turning_bias), std), max), min) in bearing_hist
            .iter()
            .zip(turning_bias_hist.iter())
            .zip(error_bar_std.iter())
            .zip(error_bar_max.iter())
            .zip(error_bar_min.iter())
        {
            writeln!(
                file,
                "{}, {}, {}, {}, {}",
                bearing, turning_bias, std, max, min
            )
            .unwrap();
        }
    }

    pub fn normal_gradient(
        r_vec: &[[f64; 2]],
        constant: &Const,
        gauss_setting: &Gausssetting,
        time: &Time,
        periodic_number: &usize,
        delta: f64,
        mode: usize,
    ) -> Vec<f64> {
        let mut bearing_point: Vec<[[f64; 2]; 2]> = Vec::new();
        for i in 0..time.simulation_time - 2 * periodic_number * time.periodic_time {
            bearing_point.push([r_vec[i], r_vec[i + periodic_number * time.periodic_time]]);
        }
        let mut normal_gradient: Vec<f64> = Vec::new();

        // 法線ベクトル
        for bearing_point_item in &bearing_point {
            let bearing_vec: [f64; 2] = [
                -bearing_point_item[1][1] + bearing_point_item[0][1],
                bearing_point_item[1][0] - bearing_point_item[0][0],
            ];

            let magnitude_bearing: f64 = (bearing_vec[0].powi(2) + bearing_vec[1].powi(2)).sqrt();

            let normal_gradient_delta: [f64; 2] = [
                bearing_vec[0] / magnitude_bearing * delta,
                bearing_vec[1] / magnitude_bearing * delta,
            ];

            let mut normal: f64 = 0.0;
            if mode == 0 {
                normal = (concentration(
                    constant,
                    bearing_point_item[0][0] + normal_gradient_delta[0],
                    bearing_point_item[0][1] + normal_gradient_delta[1],
                ) - concentration(
                    constant,
                    bearing_point_item[0][0],
                    bearing_point_item[0][1],
                )) / delta;
            } else if mode == 1 {
                normal = (gauss_concentration(
                    gauss_setting,
                    constant,
                    bearing_point_item[0][0] + normal_gradient_delta[0],
                    bearing_point_item[0][1] + normal_gradient_delta[1],
                ) - gauss_concentration(
                    gauss_setting,
                    constant,
                    bearing_point_item[0][0],
                    bearing_point_item[0][1],
                )) / delta;
            } else if mode == 2 {
                normal = (two_gauss_concentration(
                    gauss_setting,
                    constant,
                    bearing_point_item[0][0] + normal_gradient_delta[0],
                    bearing_point_item[0][1] + normal_gradient_delta[1],
                ) - two_gauss_concentration(
                    gauss_setting,
                    constant,
                    bearing_point_item[0][0],
                    bearing_point_item[0][1],
                )) / delta;
            }
            normal_gradient.push(normal);
        }

        normal_gradient
    }

    pub fn histgram_count_normal_gradient(
        nomal_gradient: &[f64],
        turning_bias: &[f64],
        bin_number: usize,
        concentration_gradient_max: f64,
    ) -> Vec<f64> {
        //ヒストグラムの種
        let mut turning_bias_mean: Vec<f64> = Vec::new();
        let step_size: f64 = 2.0 * concentration_gradient_max / bin_number as f64;

        for i in 0..bin_number {
            let mut mean: Vec<f64> = Vec::new();
            for (j, &nomal_gradient_value) in nomal_gradient.iter().enumerate() {
                if ((-concentration_gradient_max + (i as f64) * step_size) < nomal_gradient_value)
                    && (nomal_gradient_value
                        < (-concentration_gradient_max + ((i + 1) as f64) * step_size))
                {
                    mean.push(turning_bias[j]);
                }
            }
            // 平均値の計算
            let mean_value: f64 = if !mean.is_empty() {
                mean.iter().sum::<f64>() / mean.len() as f64
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };
            turning_bias_mean.push(mean_value);
        }

        turning_bias_mean
    }

    pub fn analysis_klinotaxis_nomal_gradient_errbar_std_max_min(
        result_ga: &[Ga],
        file_name: &Filename,
        liner_setting: &Setting,
        gauss_setting: &Gausssetting,
        analysis_setting: &Analysissetting,
        analysis_use_function: &i32,
    ) {
        let hist_mean: Vec<Vec<f64>> = (0..analysis_setting.analysis_loop)
            .into_par_iter()
            .map(|_| {
                let result: Vec<Vec<f64>> = klinotaxis_analysis(
                    &result_ga[analysis_setting.gene_number].gene,
                    liner_setting,
                    gauss_setting,
                    analysis_setting.mode,
                    analysis_setting.periodic_number,
                    analysis_setting.periodic_number_drain,
                    analysis_setting.delta,
                    analysis_use_function,
                );

                histgram_count_normal_gradient(
                    &result[0],
                    &result[1],
                    analysis_setting.bin_number,
                    analysis_setting.concentration_gradient_max,
                )
            })
            .collect::<Vec<Vec<f64>>>();

        // ヒストグラムの作成
        let mut normal_gradient_hist: Vec<f64> = Vec::new();
        let mut turning_bias_hist: Vec<f64> = Vec::new();
        let mut error_bar_std: Vec<f64> = Vec::new();
        let mut error_bar_max: Vec<f64> = Vec::new();
        let mut error_bar_min: Vec<f64> = Vec::new();

        let step_size: f64 =
            2.0 * analysis_setting.concentration_gradient_max / analysis_setting.bin_number as f64;

        for i in 0..analysis_setting.bin_number {
            let mut mean: Vec<f64> = Vec::new();
            for row in &hist_mean {
                if !row[i].is_nan() {
                    mean.push(row[i]);
                }
            }

            // 平均値の計算
            let mean_value: f64 = if !mean.is_empty() {
                mean.iter().sum::<f64>() / mean.len() as f64
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };

            // エラーバーの計算
            let max: f64 = if !mean.is_empty() {
                mean.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            } else {
                f64::NAN
            };

            let min: f64 = if !mean.is_empty() {
                mean.iter().cloned().fold(f64::INFINITY, f64::min)
            } else {
                f64::NAN
            };

            // 標準偏差の計算
            let std: f64 = if !mean.is_empty() {
                let variance: f64 =
                    mean.iter().map(|&x| (x - mean_value).powi(2)).sum::<f64>() / mean.len() as f64;
                variance.sqrt()
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };

            normal_gradient_hist
                .push(-analysis_setting.concentration_gradient_max + (i as f64) * step_size);
            turning_bias_hist.push(mean_value);
            error_bar_std.push(std);
            error_bar_max.push(max);
            error_bar_min.push(min);
        }

        // Open a file for writing
        let mut file: File =
            File::create(&file_name.nomal_gradient_vs_turning_bias_output).unwrap();

        // Iterate over the vectors and write each triplet of values to a line in the file
        for ((((nomal, turning_bias), std), max), min) in normal_gradient_hist
            .iter()
            .zip(turning_bias_hist.iter())
            .zip(error_bar_std.iter())
            .zip(error_bar_max.iter())
            .zip(error_bar_min.iter())
        {
            writeln!(
                file,
                "{}, {}, {}, {}, {}",
                nomal, turning_bias, std, max, min
            )
            .unwrap();
        }
    }

    pub fn translational_gradient(
        r_vec: &[[f64; 2]],
        constant: &Const,
        gauss_setting: &Gausssetting,
        time: &Time,
        periodic_number: &usize,
        delta: f64,
        mode: usize,
    ) -> Vec<f64> {
        let mut bearing_point: Vec<[[f64; 2]; 2]> = Vec::new();
        for i in 0..time.simulation_time - 2 * periodic_number * time.periodic_time {
            bearing_point.push([r_vec[i], r_vec[i + periodic_number * time.periodic_time]]);
        }
        let mut translational_gradient: Vec<f64> = Vec::new();

        // 並進方向のベクトル
        for bearing_point_item in &bearing_point {
            let bearing_vec: [f64; 2] = [
                bearing_point_item[1][0] - bearing_point_item[0][0],
                bearing_point_item[1][1] - bearing_point_item[0][1],
            ];

            let magnitude_bearing: f64 = (bearing_vec[0].powi(2) + bearing_vec[1].powi(2)).sqrt();

            let translational_gradient_delta: [f64; 2] = [
                bearing_vec[0] / magnitude_bearing * delta,
                bearing_vec[1] / magnitude_bearing * delta,
            ];

            let mut translational: f64 = 0.0;
            if mode == 0 {
                translational = (concentration(
                    constant,
                    bearing_point_item[0][0] + translational_gradient_delta[0],
                    bearing_point_item[0][1] + translational_gradient_delta[1],
                ) - concentration(
                    constant,
                    bearing_point_item[0][0],
                    bearing_point_item[0][1],
                )) / delta;
            } else if mode == 1 {
                translational = (gauss_concentration(
                    gauss_setting,
                    constant,
                    bearing_point_item[0][0] + translational_gradient_delta[0],
                    bearing_point_item[0][1] + translational_gradient_delta[1],
                ) - gauss_concentration(
                    gauss_setting,
                    constant,
                    bearing_point_item[0][0],
                    bearing_point_item[0][1],
                )) / delta;
            } else if mode == 2 {
                translational = (two_gauss_concentration(
                    gauss_setting,
                    constant,
                    bearing_point_item[0][0] + translational_gradient_delta[0],
                    bearing_point_item[0][1] + translational_gradient_delta[1],
                ) - two_gauss_concentration(
                    gauss_setting,
                    constant,
                    bearing_point_item[0][0],
                    bearing_point_item[0][1],
                )) / delta;
            }

            translational_gradient.push(translational);
        }

        translational_gradient
    }

    pub fn histgram_count_translational_gradient(
        translational_gradient: &[f64],
        turning_bias: &[f64],
        bin_number: usize,
        concentration_gradient_max: f64,
    ) -> Vec<Vec<f64>> {
        // ヒストグラムの種
        let mut turning_bias_mean: Vec<f64> = Vec::new();
        let mut positive_mean: Vec<f64> = Vec::new();
        let mut negative_mean: Vec<f64> = Vec::new();
        let step_size: f64 = 2.0 * concentration_gradient_max / bin_number as f64;

        for i in 0..bin_number {
            let mut mean: Vec<f64> = Vec::new();

            for (j, &translational_gradient_value) in translational_gradient.iter().enumerate() {
                if ((-concentration_gradient_max + (i as f64) * step_size)
                    < translational_gradient_value)
                    && (translational_gradient_value
                        < (-concentration_gradient_max + ((i + 1) as f64) * step_size))
                {
                    mean.push(turning_bias[j]);
                }
            }

            // 平均値の計算
            let mean_value: f64 = if !mean.is_empty() {
                mean.iter().sum::<f64>() / mean.len() as f64
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };

            turning_bias_mean.push(mean_value);

            // 正の値のみの平均
            let positive_values: Vec<f64> = mean.iter().cloned().filter(|&x| x > 0.0).collect();
            let positive_mean_value: f64 = if !positive_values.is_empty() {
                positive_values.iter().sum::<f64>() / positive_values.len() as f64
            } else {
                f64::NAN
            };
            positive_mean.push(positive_mean_value);

            // 負の値のみの平均
            let negative_values: Vec<f64> = mean.iter().cloned().filter(|&x| x < 0.0).collect();
            let negative_mean_value: f64 = if !negative_values.is_empty() {
                negative_values.iter().sum::<f64>() / negative_values.len() as f64
            } else {
                f64::NAN
            };
            negative_mean.push(negative_mean_value);
        }

        // 結果を返す
        let result: Vec<Vec<f64>> = vec![turning_bias_mean, positive_mean, negative_mean];

        result
    }

    pub fn analysis_klinotaxis_translational_gradient_errbar_std_positive_negative(
        result_ga: &[Ga],
        file_name: &Filename,
        liner_setting: &Setting,
        gauss_setting: &Gausssetting,
        analysis_setting: &Analysissetting,
        analysis_use_function: &i32,
    ) {
        let hist_mean: Vec<Vec<Vec<f64>>> = (0..analysis_setting.analysis_loop)
            .into_par_iter()
            .map(|_| {
                let result: Vec<Vec<f64>> = klinotaxis_analysis(
                    &result_ga[analysis_setting.gene_number].gene,
                    liner_setting,
                    gauss_setting,
                    analysis_setting.mode,
                    analysis_setting.periodic_number,
                    analysis_setting.periodic_number_drain,
                    analysis_setting.delta,
                    analysis_use_function,
                );

                histgram_count_translational_gradient(
                    &result[0],
                    &result[1],
                    analysis_setting.bin_number,
                    analysis_setting.concentration_gradient_max,
                )
            })
            .collect::<Vec<Vec<Vec<f64>>>>();

        // ヒストグラムの作成
        let mut translational_gradient_hist: Vec<f64> = Vec::new();
        let mut turning_bias_hist: Vec<f64> = Vec::new();
        let mut turning_bias_positive_hist: Vec<f64> = Vec::new();
        let mut turning_bias_negative_hist: Vec<f64> = Vec::new();
        let mut error_bar_std: Vec<f64> = Vec::new();
        let mut error_bar_positive_std: Vec<f64> = Vec::new();
        let mut error_bar_negative_std: Vec<f64> = Vec::new();

        let step_size: f64 =
            2.0 * analysis_setting.concentration_gradient_max / analysis_setting.bin_number as f64;

        for i in 0..analysis_setting.bin_number {
            let mut mean: Vec<f64> = Vec::new();
            let mut positive_mean: Vec<f64> = Vec::new();
            let mut negative_mean: Vec<f64> = Vec::new();

            for row in &hist_mean {
                if !row[0][i].is_nan() {
                    mean.push(row[0][i]);
                }
            }

            for row in &hist_mean {
                if !row[1][i].is_nan() {
                    positive_mean.push(row[1][i]);
                }
            }

            for row in &hist_mean {
                if !row[2][i].is_nan() {
                    negative_mean.push(row[2][i]);
                }
            }

            // 平均値の計算
            let mean_value: f64 = if !mean.is_empty() {
                mean.iter().sum::<f64>() / mean.len() as f64
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };

            let positive_mean_value: f64 = if !positive_mean.is_empty() {
                positive_mean.iter().sum::<f64>() / positive_mean.len() as f64
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };

            let negative_mean_value: f64 = if !negative_mean.is_empty() {
                negative_mean.iter().sum::<f64>() / negative_mean.len() as f64
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };

            // 標準偏差の計算
            let std: f64 = if !mean.is_empty() {
                let variance: f64 =
                    mean.iter().map(|&x| (x - mean_value).powi(2)).sum::<f64>() / mean.len() as f64;
                variance.sqrt()
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };

            let positive_std: f64 = if !positive_mean.is_empty() {
                let variance: f64 = positive_mean
                    .iter()
                    .map(|&x| (x - positive_mean_value).powi(2))
                    .sum::<f64>()
                    / positive_mean.len() as f64;
                variance.sqrt()
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };

            let negative_std: f64 = if !negative_mean.is_empty() {
                let variance: f64 = negative_mean
                    .iter()
                    .map(|&x| (x - negative_mean_value).powi(2))
                    .sum::<f64>()
                    / negative_mean.len() as f64;
                variance.sqrt()
            } else {
                f64::NAN // もし mean が空なら NaN をセット
            };

            translational_gradient_hist
                .push(-analysis_setting.concentration_gradient_max + (i as f64) * step_size);
            turning_bias_hist.push(mean_value);
            turning_bias_positive_hist.push(positive_mean_value);
            turning_bias_negative_hist.push(negative_mean_value);
            error_bar_std.push(std);
            error_bar_positive_std.push(positive_std);
            error_bar_negative_std.push(negative_std);
        }

        // Open a file for writing
        let mut file: File =
            File::create(&file_name.translational_gradient_vs_turning_bias_output).unwrap();

        // Iterate over the vectors and write each triplet of values to a line in the file
        for (
            (
                ((((translational, turning_bias), std), positive_turning_bias), positive_std),
                negative_turning_bias,
            ),
            negative_std,
        ) in translational_gradient_hist
            .iter()
            .zip(turning_bias_hist.iter())
            .zip(error_bar_std.iter())
            .zip(turning_bias_positive_hist.iter())
            .zip(error_bar_positive_std.iter())
            .zip(turning_bias_negative_hist.iter())
            .zip(error_bar_negative_std.iter())
        {
            writeln!(
                file,
                "{}, {}, {}, {}, {}, {}, {}",
                translational,
                turning_bias,
                std,
                positive_turning_bias,
                positive_std,
                negative_turning_bias,
                negative_std,
            )
            .unwrap();
        }
    }
    pub fn analysis() {
        // Result.jsonを読み込む
        let result_ga: Vec<Ga> = read_result();

        // klinotaxis_analysis.toml ファイルを読み込む
        let toml_str: String =
            std::fs::read_to_string("klinotaxis_analysis.toml").expect("Failed to read file");
        let value: Value = toml::from_str(&toml_str).expect("Failed to parse TOML");
        let file_name_base: Filename = value["file_name"]
            .clone()
            .try_into()
            .expect("Failed to parse Setting");
        let mut file_name: Filename = value["file_name"]
            .clone()
            .try_into()
            .expect("Failed to parse Setting");
        let liner_setting: Setting = value["liner_setting"]
            .clone()
            .try_into()
            .expect("Failed to parse Setting");
        let gauss_setting: Gausssetting = value["gauss_setting"]
            .clone()
            .try_into()
            .expect("Failed to parse Setting");
        let mut analysis_setting: Analysissetting = value["analysis_setting"]
            .clone()
            .try_into()
            .expect("Failed to parse Setting");
        let analysis_use_gene: Analysisusegene = value["analysis_use_gene"]
            .clone()
            .try_into()
            .expect("Failed to parse Setting");
        let analysis_use_function: Analysisusefunction = value["analysis_use_function"]
            .clone()
            .try_into()
            .expect("Failed to parse Setting");

        // 分析する遺伝子の指定
        let mut gene_range: Vec<usize> = Vec::new();
        if analysis_use_gene.mode == 0 {
            gene_range = analysis_use_gene.gene_numbers
        } else if analysis_use_gene.mode == 1 {
            gene_range = (analysis_use_gene.gene_number_range[0]
                ..analysis_use_gene.gene_number_range[1] + 1)
                .collect()
        }

        // 各遺伝子での分析のループ
        for i in gene_range {
            // 遺伝子番号の変更
            analysis_setting.gene_number = i;
            // ファイルネームの変更
            file_name.bearing_vs_turning_bias_output = format!(
                "{}_{}{}",
                file_name_base.bearing_vs_turning_bias_output, i, ".txt"
            );
            file_name.nomal_gradient_vs_turning_bias_output = format!(
                "{}_{}{}",
                file_name_base.nomal_gradient_vs_turning_bias_output, i, ".txt"
            );
            file_name.translational_gradient_vs_turning_bias_output = format!(
                "{}_{}{}",
                file_name_base.translational_gradient_vs_turning_bias_output, i, ".txt"
            );
            // 分析方法のループ
            for j in &analysis_use_function.mode {
                if *j == 0 {
                    // turning bias vs bearing
                    analysis_klinotaxis_bearing_errbar_std_max_min(
                        &result_ga,
                        &file_name,
                        &liner_setting,
                        &gauss_setting,
                        &analysis_setting,
                        j,
                    );
                } else if *j == 1 {
                    // turning bias vs nomal gradient
                    analysis_klinotaxis_nomal_gradient_errbar_std_max_min(
                        &result_ga,
                        &file_name,
                        &liner_setting,
                        &gauss_setting,
                        &analysis_setting,
                        j,
                    );
                } else if *j == 2 {
                    // turning bias vs translational gradient
                    analysis_klinotaxis_translational_gradient_errbar_std_positive_negative(
                        &result_ga,
                        &file_name,
                        &liner_setting,
                        &gauss_setting,
                        &analysis_setting,
                        j,
                    );
                }
            }
        }
    }
}
