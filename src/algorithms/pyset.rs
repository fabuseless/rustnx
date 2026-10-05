//! A replica of CPython's `set` hash table, for results whose order
//! depends on a set's iteration order (`flow`: `next(iter(s))` picks;
//! `measures`: `common_neighbors`). The Python side checks once, with
//! `replay_sets`, that it agrees with the running interpreter.

const EMPTY: u32 = u32::MAX;
const DUMMY: u32 = u32::MAX - 1;
const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;
const SET_MINSIZE: usize = 8;

/// A replica of CPython's `set` hash table (Objects/setobject.c, the same
/// in 3.10 to 3.14) holding node positions, given each node's `hash()`.
/// NetworkX picks elements with `next(iter(s))`, which returns the first
/// occupied slot; that depends on the table layout, which this reproduces
/// operation by operation (probing, dummies, resizes, `update`, `clear`).
/// Distinct nodes never compare equal, as for dict keys.
#[derive(Clone)]
pub struct PySet {
    keys: Vec<u32>,
    fill: usize,
    used: usize,
    /// Where `set.pop` starts looking (`so->finger`).
    finger: usize,
}

impl Default for PySet {
    fn default() -> Self {
        PySet {
            keys: vec![EMPTY; SET_MINSIZE],
            fill: 0,
            used: 0,
            finger: 0,
        }
    }
}

impl PySet {
    #[inline]
    fn mask(&self) -> usize {
        self.keys.len() - 1
    }

    #[inline]
    fn next_index(i: usize, perturb: &mut u64, mask: usize) -> usize {
        *perturb >>= PERTURB_SHIFT;
        (i.wrapping_mul(5)
            .wrapping_add(1)
            .wrapping_add(*perturb as usize))
            & mask
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    pub fn len(&self) -> usize {
        self.used
    }

    /// The slot holding `key`, if present.
    fn find(&self, key: u32, hashes: &[i64]) -> Option<usize> {
        let h = hashes[key as usize];
        let mask = self.mask();
        let mut perturb = h as u64;
        let mut i = (h as u64 as usize) & mask;
        loop {
            let probes = if i + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            for j in 0..=probes {
                let k = self.keys[i + j];
                if k == EMPTY {
                    return None;
                }
                if k == key {
                    return Some(i + j);
                }
            }
            i = Self::next_index(i, &mut perturb, mask);
        }
    }

    pub fn contains(&self, key: u32, hashes: &[i64]) -> bool {
        self.find(key, hashes).is_some()
    }

    /// `set_add_entry`.
    pub fn add(&mut self, key: u32, hashes: &[i64]) {
        let h = hashes[key as usize];
        let mask = self.mask();
        let mut perturb = h as u64;
        let mut i = (h as u64 as usize) & mask;
        let mut freeslot = None;
        loop {
            let probes = if i + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            for j in 0..=probes {
                let idx = i + j;
                let k = self.keys[idx];
                if k == EMPTY {
                    if let Some(slot) = freeslot {
                        self.used += 1;
                        self.keys[slot] = key;
                        return;
                    }
                    self.fill += 1;
                    self.used += 1;
                    self.keys[idx] = key;
                    if self.fill * 5 < mask * 3 {
                        return;
                    }
                    let minused = if self.used > 50000 {
                        self.used * 2
                    } else {
                        self.used * 4
                    };
                    self.resize(minused, hashes);
                    return;
                }
                if k == key {
                    return;
                }
                if k == DUMMY {
                    // CPython keeps the last dummy seen on the probe path.
                    freeslot = Some(idx);
                }
            }
            i = Self::next_index(i, &mut perturb, mask);
        }
    }

    /// `set_insert_clean`: `key` is absent and the table has no dummies.
    fn insert_clean(&mut self, key: u32, hashes: &[i64]) {
        let h = hashes[key as usize];
        let mask = self.mask();
        let mut perturb = h as u64;
        let mut i = (h as u64 as usize) & mask;
        loop {
            if self.keys[i] == EMPTY {
                self.keys[i] = key;
                return;
            }
            if i + LINEAR_PROBES <= mask {
                for j in 1..=LINEAR_PROBES {
                    if self.keys[i + j] == EMPTY {
                        self.keys[i + j] = key;
                        return;
                    }
                }
            }
            i = Self::next_index(i, &mut perturb, mask);
        }
    }

    /// `set_table_resize`.
    fn resize(&mut self, minused: usize, hashes: &[i64]) {
        let mut newsize = SET_MINSIZE;
        while newsize <= minused {
            newsize <<= 1;
        }
        if newsize == SET_MINSIZE && self.keys.len() == SET_MINSIZE && self.fill == self.used {
            return;
        }
        let old = std::mem::replace(&mut self.keys, vec![EMPTY; newsize]);
        self.fill = self.used;
        for k in old {
            if k != EMPTY && k != DUMMY {
                self.insert_clean(k, hashes);
            }
        }
    }

    /// `set.discard`; whether `key` was present (`set.remove` raises if not).
    pub fn discard(&mut self, key: u32, hashes: &[i64]) -> bool {
        match self.find(key, hashes) {
            Some(idx) => {
                self.keys[idx] = DUMMY;
                self.used -= 1;
                true
            }
            None => false,
        }
    }

    /// `set(d)` for a dict `d` with these keys, in order (`set_update_internal`
    /// presizes the table for the dict before adding its keys).
    pub fn from_dict(keys: &[u32], hashes: &[i64]) -> Self {
        let mut s = PySet::default();
        if (s.fill + keys.len()) * 5 >= s.mask() * 3 {
            s.resize((s.used + keys.len()) * 2, hashes);
        }
        for &k in keys {
            s.add(k, hashes);
        }
        s
    }

    /// The end of `set_difference_update_internal`: purge many dummies.
    pub fn after_difference_update(&mut self, hashes: &[i64]) {
        if self.fill - self.used > self.mask() / 4 {
            let minused = if self.used > 50000 {
                self.used * 2
            } else {
                self.used * 4
            };
            self.resize(minused, hashes);
        }
    }

    pub fn clear(&mut self) {
        self.keys = vec![EMPTY; SET_MINSIZE];
        self.fill = 0;
        self.used = 0;
    }

    /// `next(iter(s))`.
    pub fn first(&self) -> Option<u32> {
        self.keys
            .iter()
            .copied()
            .find(|&k| k != EMPTY && k != DUMMY)
    }

    /// The elements in iteration order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.keys
            .iter()
            .copied()
            .filter(|&k| k != EMPTY && k != DUMMY)
    }

    /// `self.update(other)` with `other` a set (`set_merge`).
    pub fn merge(&mut self, other: &PySet, hashes: &[i64]) {
        if other.used == 0 {
            return;
        }
        if (self.fill + other.used) * 5 >= self.mask() * 3 {
            self.resize((self.used + other.used) * 2, hashes);
        }
        if self.fill == 0 && self.mask() == other.mask() && other.fill == other.used {
            for (i, &k) in other.keys.iter().enumerate() {
                if k != EMPTY {
                    self.keys[i] = k;
                }
            }
            self.fill = other.fill;
            self.used = other.used;
            return;
        }
        if self.fill == 0 {
            self.fill = other.used;
            self.used = other.used;
            for k in other.iter() {
                self.insert_clean(k, hashes);
            }
            return;
        }
        for k in other.iter() {
            self.add(k, hashes);
        }
    }
}

/// `sets[dst].update(sets[src])` for two sets in one vector.
pub fn merge_sets(sets: &mut [PySet], dst: usize, src: usize, hashes: &[i64]) {
    if dst == src {
        return; // a.update(a) does nothing
    }
    if dst < src {
        let (a, b) = sets.split_at_mut(src);
        a[dst].merge(&b[0], hashes);
    } else {
        let (a, b) = sets.split_at_mut(dst);
        b[0].merge(&a[src], hashes);
    }
}

/// Replays set operations for the runtime check that `PySet` matches the
/// running interpreter: `(op, set, key)` with op 0 add, 1 discard, 2 clear,
/// 3 update set from set `key`; batch 25's operations: 4 pop (records the
/// popped element), 5 `s = s & k`, 6 `s = s - k`, 7 `s = s.copy()`, 8
/// `s -= k`, 9 `s &= k`, 10 `s = s | k`. After each op, the first element of the
/// set operated on (or -1); at the end, every set's elements in order.
pub fn replay_sets(
    hashes: &[i64],
    nsets: usize,
    ops: &[(u8, u32, u32)],
) -> (Vec<i64>, Vec<Vec<u32>>) {
    let mut sets = vec![PySet::default(); nsets];
    let mut firsts = Vec::with_capacity(ops.len());
    for &(op, s, k) in ops {
        let s = s as usize;
        match op {
            0 => sets[s].add(k, hashes),
            1 => {
                sets[s].discard(k, hashes);
            }
            2 => sets[s].clear(),
            3 => merge_sets(&mut sets, s, k as usize, hashes),
            4 => {
                firsts.push(sets[s].pop().map_or(-1, |v| v as i64));
                continue;
            }
            5 => sets[s] = sets[s].intersection(&sets[k as usize], hashes),
            6 => sets[s] = sets[s].difference(&sets[k as usize], hashes),
            7 => sets[s] = sets[s].copy(hashes),
            8 => {
                if s == k as usize {
                    sets[s].clear(); // `set_clear_internal` leaves the finger
                } else {
                    let other = sets[k as usize].clone();
                    sets[s].difference_update(&other, hashes);
                }
            }
            9 => {
                let finger = sets[s].finger;
                sets[s] = sets[s].intersection(&sets[k as usize], hashes);
                sets[s].finger = finger;
            }
            _ => sets[s] = sets[s].union(&sets[k as usize], hashes),
        }
        firsts.push(sets[s].first().map_or(-1, |v| v as i64));
    }
    (firsts, sets.iter().map(|s| s.iter().collect()).collect())
}

// --- Batch 25: set algebra (`&`, `-`, `|`, `copy`, `pop`, ...) ---

impl PySet {
    /// `set(iterable)` for an iterable that is neither a set nor an exact
    /// dict (a list, a generator, a graph, an `AtlasView`): one `add` each.
    pub fn from_iter(keys: impl IntoIterator<Item = u32>, hashes: &[i64]) -> Self {
        let mut s = PySet::default();
        for k in keys {
            s.add(k, hashes);
        }
        s
    }

    /// `s.copy()` / `set(s)` (`set_merge` into a new set).
    pub fn copy(&self, hashes: &[i64]) -> Self {
        let mut s = PySet::default();
        s.merge(self, hashes);
        s
    }

    /// `self & other` (`set_intersection`): iterates the smaller set (the
    /// right operand on a tie), adding the elements the other holds.
    pub fn intersection(&self, other: &PySet, hashes: &[i64]) -> Self {
        if std::ptr::eq(self, other) {
            return self.copy(hashes);
        }
        let (small, big) = if other.used > self.used {
            (self, other)
        } else {
            (other, self)
        };
        let mut s = PySet::default();
        for k in small.iter() {
            if big.contains(k, hashes) {
                s.add(k, hashes);
            }
        }
        s
    }

    /// `self &= other`: the intersection's table, keeping this set's
    /// `pop` finger (`set_swap_bodies` leaves it).
    pub fn intersection_update(&mut self, other: &PySet, hashes: &[i64]) {
        let finger = self.finger;
        *self = self.intersection(other, hashes);
        self.finger = finger;
    }

    /// `self - other` (`set_difference`) with `other` a set: a copy with
    /// `other`'s elements discarded if this set is over four times larger,
    /// else the elements `other` lacks, added in order to a new set.
    pub fn difference(&self, other: &PySet, hashes: &[i64]) -> Self {
        if (self.used >> 2) > other.used {
            let mut s = self.copy(hashes);
            s.difference_update(other, hashes);
            return s;
        }
        let mut s = PySet::default();
        for k in self.iter() {
            if !other.contains(k, hashes) {
                s.add(k, hashes);
            }
        }
        s
    }

    /// `self -= other` for another set (`set_difference_update_internal`).
    pub fn difference_update(&mut self, other: &PySet, hashes: &[i64]) {
        self.discard_all(other.iter(), hashes);
    }

    /// `self.difference_update(iterable)` for a non-set iterable: discard
    /// each, then purge many dummies.
    pub fn discard_all(&mut self, keys: impl IntoIterator<Item = u32>, hashes: &[i64]) {
        for k in keys {
            self.discard(k, hashes);
        }
        self.after_difference_update(hashes);
    }

    /// `self | other` (`set_union`: a copy, then `update`).
    pub fn union(&self, other: &PySet, hashes: &[i64]) -> Self {
        let mut s = self.copy(hashes);
        if !std::ptr::eq(self, other) {
            s.merge(other, hashes);
        }
        s
    }

    /// `set.pop()`: the first element at or after the finger, wrapping.
    pub fn pop(&mut self) -> Option<u32> {
        if self.used == 0 {
            return None;
        }
        let mask = self.mask();
        let mut i = self.finger & mask;
        while self.keys[i] == EMPTY || self.keys[i] == DUMMY {
            i += 1;
            if i > mask {
                i = 0;
            }
        }
        let key = self.keys[i];
        self.keys[i] = DUMMY;
        self.used -= 1;
        self.finger = i + 1;
        Some(key)
    }
}
