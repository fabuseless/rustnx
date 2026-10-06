//! CPython's `random.Random` (Modules/_randommodule.c and Lib/random.py,
//! the same in 3.10 to 3.14), for replaying a seeded NetworkX function's
//! random draws bit for bit. The Python side builds the generator from
//! `seed.getstate()` and hands the new state back with `seed.setstate`.
//!
//! Only integer arguments that fit in 64 bits are covered: Python's
//! arbitrary-size ints (`getrandbits(k)` for `k > 64`) are left to the
//! callers, which fall back to NetworkX for them.

use std::collections::HashSet;

/// CPython's Mersenne Twister (`random.Random`), from its `getstate()`.
#[derive(Clone)]
pub struct Mt19937 {
    mt: [u32; 624],
    index: usize,
}

impl Mt19937 {
    /// From the 625 ints of `getstate()[1]`.
    pub fn from_state(state: &[u32]) -> Option<Self> {
        if state.len() != 625 || state[624] > 624 {
            return None;
        }
        let mut mt = [0u32; 624];
        mt.copy_from_slice(&state[..624]);
        Some(Mt19937 {
            mt,
            index: state[624] as usize,
        })
    }

    pub fn state(&self) -> Vec<u32> {
        let mut s = self.mt.to_vec();
        s.push(self.index as u32);
        s
    }

    /// `genrand_uint32`.
    pub fn genrand_uint32(&mut self) -> u32 {
        const N: usize = 624;
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        let mag01 = [0u32, MATRIX_A];
        if self.index >= N {
            let mt = &mut self.mt;
            for kk in 0..N - M {
                let y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            for kk in N - M..N - 1 {
                let y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            let y = (mt[N - 1] & UPPER) | (mt[0] & LOWER);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `Random.random()`: 53 random bits from two draws.
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_uint32() >> 5) as f64;
        let b = (self.genrand_uint32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `Random.getrandbits(k)` for `k <= 64`. Wider results are filled
    /// with 32-bit words from the least significant end, and the last
    /// word drops its low bits.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        debug_assert!(k <= 64);
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            return (self.genrand_uint32() >> (32 - k)) as u64;
        }
        let low = self.genrand_uint32() as u64;
        let high = (self.genrand_uint32() >> (64 - k)) as u64;
        (high << 32) | low
    }

    /// `Random._randbelow_with_getrandbits(n)` for `n >= 1`.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        // Python raises for 0 (an empty `choice`); the replica would spin.
        assert!(n >= 1, "randbelow(0)");
        let k = u64::BITS - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `randbelow` for an index.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        self.randbelow(n as u64) as usize
    }

    /// `Random.randrange(start, stop)`; `None` for an empty range (Python
    /// raises a `ValueError` whose text depends on the version).
    pub fn randrange(&mut self, start: i64, stop: i64) -> Option<i64> {
        let width = stop as i128 - start as i128;
        if width <= 0 || width > u64::MAX as i128 {
            return None;
        }
        Some((start as i128 + self.randbelow(width as u64) as i128) as i64)
    }

    /// `Random.randint(a, b)`.
    pub fn randint(&mut self, a: i64, b: i64) -> Option<i64> {
        self.randrange(a, b.checked_add(1)?)
    }

    /// `Random.choice` (non-empty `seq`).
    pub fn choice<T: Copy>(&mut self, seq: &[T]) -> T {
        seq[self.below(seq.len())]
    }

    /// `Random.shuffle`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.below(i + 1);
            x.swap(i, j);
        }
    }

    /// `Random.uniform(a, b)`.
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }

    /// `Random.paretovariate(alpha)`: `(1.0 - random()) ** (-1.0 / alpha)`.
    /// Python raises where `pow` overflows (an infinite result here) and
    /// for `alpha == 0`; callers check both.
    pub fn paretovariate(&mut self, alpha: f64) -> f64 {
        let u = 1.0 - self.random();
        u.powf(-1.0 / alpha)
    }

    /// `Random.sample(population, k)` for `k <= len(population)`: a pool
    /// for small populations, else rejection against the picks so far.
    pub fn sample<T: Copy>(&mut self, population: &[T], k: usize) -> Vec<T> {
        self.sample_range(population.len(), k)
            .into_iter()
            .map(|j| population[j])
            .collect()
    }

    /// `Random.sample(range(n), k)` for `k <= n`: the positions `sample`
    /// picks from a population of length `n`, without building it.
    pub fn sample_range(&mut self, n: usize, k: usize) -> Vec<usize> {
        assert!(k <= n, "sample larger than population");
        let mut setsize: u128 = 21;
        if k > 5 {
            // `4 ** _ceil(_log(k * 3, 4))`, with math.log's float division.
            let exponent = (((k as f64) * 3.0).ln() / 4f64.ln()).ceil() as u32;
            setsize = setsize.saturating_add(4u128.saturating_pow(exponent));
        }
        let mut result = Vec::with_capacity(k);
        if n as u128 <= setsize {
            let mut pool: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = self.below(n - i);
                result.push(pool[j]);
                pool[j] = pool[n - i - 1];
            }
        } else {
            let mut selected = HashSet::with_capacity(k);
            for _ in 0..k {
                let mut j = self.below(n);
                while selected.contains(&j) {
                    j = self.below(n);
                }
                selected.insert(j);
                result.push(j);
            }
        }
        result
    }
}

/// Replays `random.Random` calls for the tests: `(op, a, b)` with op 0
/// `random()`, 1 `getrandbits(a)`, 2 `_randbelow(a)`, 3 `randrange(a, b)`,
/// 4 `randint(a, b)`, 5 `choice(range(a))`, 6 `uniform(a, b)` (as
/// floats), 7 `sample(range(a), b)`, 8 `shuffle(list(range(a)))`, 9
/// `paretovariate(a / b)` (as floats).
/// Each op's results come back as floats or ints; `None` for an op whose
/// arguments the replica doesn't cover.
pub enum Draw {
    Float(f64),
    Int(i128),
    Ints(Vec<u64>),
    Unsupported,
}

pub fn replay(rng: &mut Mt19937, ops: &[(u8, i64, i64)]) -> Vec<Draw> {
    ops.iter()
        .map(|&(op, a, b)| match op {
            0 => Draw::Float(rng.random()),
            1 if (0..=64).contains(&a) => Draw::Int(rng.getrandbits(a as u32) as i128),
            2 if a >= 1 => Draw::Int(rng.randbelow(a as u64) as i128),
            3 => rng
                .randrange(a, b)
                .map_or(Draw::Unsupported, |x| Draw::Int(x as i128)),
            4 => rng
                .randint(a, b)
                .map_or(Draw::Unsupported, |x| Draw::Int(x as i128)),
            5 if a >= 1 => Draw::Int(rng.randbelow(a as u64) as i128),
            6 => Draw::Float(rng.uniform(a as f64, b as f64)),
            7 if a >= 0 && b >= 0 && b <= a && a <= 1 << 24 => {
                let pop: Vec<u64> = (0..a as u64).collect();
                Draw::Ints(rng.sample(&pop, b as usize))
            }
            8 if (0..=1 << 24).contains(&a) => {
                let mut x: Vec<u64> = (0..a as u64).collect();
                rng.shuffle(&mut x);
                Draw::Ints(x)
            }
            9 if a != 0 && b != 0 => Draw::Float(rng.paretovariate(a as f64 / b as f64)),
            _ => Draw::Unsupported,
        })
        .collect()
}

// --- Batch 23 additions ---

impl Mt19937 {
    /// `Random.expovariate(lambd)`: `-log(1.0 - random()) / lambd` (the
    /// same in 3.10 to 3.14).
    pub fn expovariate(&mut self, lambd: f64) -> f64 {
        -(1.0 - self.random()).ln() / lambd
    }

    /// NumPy's `random_interval(max)` (legacy `RandomState`, whose
    /// `shuffle` and `permutation` use it) for `max < 2^32`: masked 32-bit
    /// draws until one is at most `max`.
    pub fn np_interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let value = self.genrand_uint32() & mask;
            if value <= max {
                return value;
            }
        }
    }

    /// NumPy's legacy `RandomState.permutation(n)` for `n < 2^32`: `arange(n)`
    /// shuffled from the end.
    pub fn np_permutation(&mut self, n: usize) -> Vec<u32> {
        let mut x: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            let j = self.np_interval(i as u32) as usize;
            x.swap(i, j);
        }
        x
    }
}
