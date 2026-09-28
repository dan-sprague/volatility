use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use sciplot::prelude::*;

fn draw_shocks(shocks: &mut [f64], z: &Normal<f64>, rng: &mut StdRng) {
    for e in shocks.iter_mut() {
        *e = z.sample(rng);
    }
}

fn apply(log_w: &mut [f64], shocks: &[f64], f: f64, mu: f64, sigma: f64, dt: f64) {
    let drift = (f * mu - 0.5 * f * f * sigma * sigma) * dt; // g(f) dt
    let scale = f * sigma * dt.sqrt();
    for (lw, e) in log_w.iter_mut().zip(shocks) {
        *lw += drift + scale * e;
    }
}

/// Quantiles of `data` for each q in `qs`. Leaves `data` untouched (works on a copy).
fn quantiles(data: &[f64], qs: &[f64]) -> Vec<f64> {
    let mut v = data.to_vec();
    qs.iter()
        .map(|&q| {
            let idx = ((v.len() - 1) as f64 * q).round() as usize;
            *v.select_nth_unstable_by(idx, |a, b| a.partial_cmp(b).unwrap()).1
        })
        .collect()
}

/// Mean wealth from log wealth. Unlike quantiles, the mean must be taken after exp (Jensen).
fn mean_wealth(log_w: &[f64]) -> f64 {
    log_w.iter().map(|lw| lw.exp()).sum::<f64>() / log_w.len() as f64
}

/// Percentage of people whose wealth ended below where it started.
fn percent_losing(log_w: &[f64]) -> f64 {
    100.0 * log_w.iter().filter(|&&lw| lw < 0.0).count() as f64 / log_w.len() as f64
}

fn main() {
    let mu = 0.5; // expected return per year (crypto-like)
    let sigma = 1.2; // volatility per year
    let steps_per_year = 250;
    let years = 5.0;
    let people = 10_000;
    let n_samples = 30; // example paths drawn in gray
    let seed = 42;

    let dt = 1.0 / steps_per_year as f64;
    let n_steps = (years * steps_per_year as f64) as usize;
    let z = Normal::new(0.0, 1.0).unwrap();
    let f_kelly = mu / (sigma * sigma);

    // State: one log wealth per person for each crowd, plus this step's shared shocks.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut shocks = vec![0.0; people];
    let mut log_w = vec![0.0; people]; // all in (f = 1)
    let mut log_w_kelly = vec![0.0; people]; // Kelly fraction

    // Recorded series, one value per time step.
    let mut t = Vec::with_capacity(n_steps + 1);
    let mut mean_w = Vec::with_capacity(n_steps + 1);
    let mut median_w = Vec::with_capacity(n_steps + 1);
    let mut kelly_median_w = Vec::with_capacity(n_steps + 1);
    let mut samples: Vec<Vec<f64>> = (0..n_samples).map(|_| Vec::with_capacity(n_steps + 1)).collect();

    for k in 0..=n_steps {
        if k > 0 {
            draw_shocks(&mut shocks, &z, &mut rng);
            apply(&mut log_w, &shocks, 1.0, mu, sigma, dt);
            apply(&mut log_w_kelly, &shocks, f_kelly, mu, sigma, dt);
        }

        t.push(k as f64 * dt);
        mean_w.push(mean_wealth(&log_w));
        median_w.push(quantiles(&log_w, &[0.5])[0].exp());
        kelly_median_w.push(quantiles(&log_w_kelly, &[0.5])[0].exp());
        for (path, lw) in samples.iter_mut().zip(&log_w) {
            path.push(lw.exp());
        }
    }

    // Find the richest person, then replay the same shocks to record their whole path.
    let winner = (0..people)
        .max_by(|&a, &b| log_w[a].partial_cmp(&log_w[b]).unwrap())
        .unwrap();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut replay = vec![0.0; people];
    let mut winner_w = Vec::with_capacity(n_steps + 1);
    winner_w.push(1.0);
    for _ in 0..n_steps {
        draw_shocks(&mut shocks, &z, &mut rng);
        apply(&mut replay, &shocks, 1.0, mu, sigma, dt);
        winner_w.push(replay[winner].exp());
    }
    assert_eq!(replay[winner], log_w[winner]); // the replay must match the first run exactly

    println!(
        "all in:     mean = {:.2}, median = {:.3}, lost money: {:.1}%",
        mean_w.last().unwrap(),
        median_w.last().unwrap(),
        percent_losing(&log_w)
    );
    println!(
        "kelly {f_kelly:.2}: mean = {:.2}, median = {:.3}, lost money: {:.1}%",
        mean_wealth(&log_w_kelly),
        kelly_median_w.last().unwrap(),
        percent_losing(&log_w_kelly)
    );

    std::fs::create_dir_all("out").ok();
    fig1(mu, sigma, &t, &samples, &winner_w);
    fig2(mu, sigma, &t, &samples, &winner_w, &kelly_median_w);
}

fn base_figure(mu: f64, sigma: f64, t: &[f64], samples: &[Vec<f64>], winner_w: &[f64]) -> (Figure, Axis) {
    let fig = Figure!(size = (6.5 * INCH, 3.5 * INCH), fontsize = 9.0 * PT);
    let ax = Axis!(fig.at(1, 1); xlabel = "Years", ylabel = "Wealth", yscale = Scale::Log10,
                   yminorticksvisible = true, topspinevisible = false, rightspinevisible = false);

    for path in samples {
        lines!(ax, t, path; color = (BLACK, 0.25), linewidth = 1);
    }
    lines!(ax, t, winner_w; color = RED, linewidth = 1.5, label = "Winner");
    lines!(ax, t, t.iter().map(|&t| (mu * t).exp());
           color = WONG[0], linewidth = 3, label = "All in: expected");
    lines!(ax, t, t.iter().map(|&t| ((mu - 0.5 * sigma * sigma) * t).exp());
           color = WONG[5], linewidth = 3, label = "All in: median");
    hlines!(ax, [1.0]; color = BLACK, linewidth = 1, linestyle = Linestyle::Dash);

    ax.xlims(0.0, 5.0);
    ax.ylims(1e-2, 1e4);
    (fig, ax)
}

fn fig1(mu: f64, sigma: f64, t: &[f64], samples: &[Vec<f64>], winner_w: &[f64]) {
    let (fig, ax) = base_figure(mu, sigma, t, samples, winner_w);
    Legend!(fig.at(1, 2), &ax; framevisible = false);
    fig.save_with("out/drag.png", Save::dpi(300)).ok();
}

fn fig2(mu: f64, sigma: f64, t: &[f64], samples: &[Vec<f64>], winner_w: &[f64], kelly_median_w: &[f64]) {
    let (fig, ax) = base_figure(mu, sigma, t, samples, winner_w);
    lines!(ax, t, kelly_median_w; color = WONG[2], linewidth = 3, label = "Kelly: median");
    Legend!(fig.at(1, 2), &ax; framevisible = false);
    fig.save_with("out/kelly.png", Save::dpi(300)).ok();
}
