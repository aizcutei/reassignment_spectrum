use kurbo::Rect;
use piet_common::*;
use realfft::RealFftPlanner;
use rustfft::num_complex::{Complex, ComplexFloat};
mod windows;
use rand;

fn main() {
    let test_signal = generate_test_data();
    println!("Test signal generated with {} samples", test_signal.len());

    let hann = windows::hann(2048);
    let hann_dt = windows::timederivhann(2048);
    let hann_t = windows::timeramphann(2048);

    let mut piet = Device::new().unwrap();
    let mut img = piet.bitmap_target(44 * 10, 2048, 1.0).unwrap();
    let mut piet = Device::new().unwrap();
    let mut img2 = piet.bitmap_target(44 * 10, 2048, 1.0).unwrap();
    {
        let mut ctx = img.render_context();
        let rect = Rect::new(0.0, 0.0, 44.0 * 10.0, 2048.0);
        ctx.fill(rect, &Color::rgba8(255, 255, 255, 255));
        let mut ctx = img2.render_context();
        let rect = Rect::new(0.0, 0.0, 44.0 * 10.0, 2048.0);
        ctx.fill(rect, &Color::rgba8(255, 255, 255, 255));
    }

    for i in 0..44 {
        let mut ctx = img.render_context();
        let mut ctx2 = img2.render_context();
        let sub_signal = test_signal[i * 1024..i * 1024 + 2048].to_vec();
        let mut hanned = vec![0.0; 2048];
        let mut hanned_dt = vec![0.0; 2048];
        let mut hanned_t = vec![0.0; 2048];
        for j in 0..2048 {
            hanned[j] = (sub_signal[j] * hann[j]) as f32;
            hanned_dt[j] = (sub_signal[j] * hann_dt[j]) as f32;
            hanned_t[j] = (sub_signal[j] * hann_t[j]) as f32;
        }
        let mut real_planner = RealFftPlanner::<f32>::new();
        let r2c = real_planner.plan_fft_forward(2048);
        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut hanned, &mut spectrum);
        let fft_x = spectrum.clone();
        let mag = fft_x.iter().map(|i| i.norm() / 2048.0).collect::<Vec<_>>();
        let magsqrd: Vec<_> = fft_x.iter().map(|i| i.norm_sqr()).collect();
        r2c.process(&mut hanned_dt, &mut spectrum).unwrap();
        let fft_xdt = spectrum.clone();
        r2c.process(&mut hanned_t, &mut spectrum).unwrap();
        let fft_xt = spectrum.clone();
        for j in 0..1025 {
            let rect = Rect::new(
                i as f64 * 10.0,
                (1025 - j) as f64 * 2.0,
                i as f64 * 10.0 + 10.0,
                (1025 - j) as f64 * 2.0 + 2.0,
            );
            let c = ((mag[j].log10() * 20.0 + 60.0) * 255.0 / 60.0) as u8;
            ctx2.fill(rect, &Color::rgba8(0, 0, 0, c));
            if magsqrd[j] > 0.0 {
                let freq_base = j as f32 * 24000.0 / 1025.0;
                let fc_fix = (-(fft_xdt[j] * fft_x[j].conj()).im() / magsqrd[j]);
                let fc_temp = fc_fix + freq_base;
                let tc_fix = (fft_xt[i] * fft_x[i].conj()).re() / magsqrd[i];
                let tc_temp = tc_fix + i as f32 * 1024.0 / 48000.0;
                let db = mag[j].log10() * 20.0;
                let x = 10.0 * tc_temp as f64 / 0.02133333333333333;
                let y = (1.0 - fc_temp as f64 / 24000.0) * 2048.0;
                let c = ((db + 60.0) * 255.0 / 60.0) as u8;
                let rect = Rect::new(x, y, x + 5.0, y + 1.0);
                ctx.fill(rect, &Color::rgba8(0, 0, 0, c));
            }
        }
    }
    img.save_to_file("output.png").unwrap();
    img2.save_to_file("output2.png").unwrap();
}

fn generate_test_data() -> Vec<f64> {
    let fs = 48000;
    let duration = 1.0;
    let sin_1_freq = 440.0;
    let sin_2_freq = 5000.0;

    let chirp_freq_start = 20.0;
    let chirp_freq_end = 20000.0;
    let chirp_duration = 1.0;
    let chirp_amplitude = 0.5;
    let chirp_phase = 0.0;

    let mut data = Vec::new();
    for i in 0..(fs as f64 * duration) as usize {
        let t = i as f64 / fs as f64;
        let sin_1_val = (2.0 * std::f64::consts::PI * sin_1_freq * t).sin();
        let sin_2_val = (2.0 * std::f64::consts::PI * sin_2_freq * t).sin();
        let chirp_val = chirp_amplitude
            * (2.0
                * std::f64::consts::PI
                * (chirp_freq_start + (chirp_freq_end - chirp_freq_start) * t / chirp_duration)
                * (t + chirp_phase))
                .sin();
        let noise_val = rand::random::<f64>() * 0.1;
        data.push(sin_1_val + sin_2_val + chirp_val + noise_val);
        // data.push(sin_1_val + sin_2_val);
    }
    data
}
