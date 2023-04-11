//! Dual Numbers
//!
//! Fully-featured Dual Number implementation with features for automatic differentiation of multivariate vectorial functions into gradients.
//!
//! ## Usage
//!
//! ```rust
//! extern crate dual_num;
//!
//! use dual_num::{Dual, Float, differentiate};
//!
//! fn main() {
//!     // find partial derivative at x=4.0
//!     println!("{:.5}", differentiate(4.0f64, |x| {
//!         x.sqrt() + Dual::from_real(1.0)
//!     })); // 0.25000
//! }
//! ```
//!
//! ##### Previous Work
//! * [https://github.com/FreeFull/dual_numbers](https://github.com/FreeFull/dual_numbers)
//! * [https://github.com/ibab/rust-ad](https://github.com/ibab/rust-ad)
//! * [https://github.com/tesch1/cxxduals](https://github.com/tesch1/cxxduals)

extern crate nalgebra as na;
extern crate num_traits;

use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter, LowerExp, Result as FmtResult};
use std::iter::{Product, Sum};
use std::num::FpCategory;
use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub,
    SubAssign,
};

pub use num_traits::{Float, FloatConst, Num, One, Zero};

use num_traits::{
    FromPrimitive, Inv, MulAdd, MulAddAssign, NumCast, Pow, Signed, ToPrimitive, Unsigned,
};

// Re-export traits useful for construction and extension of duals
pub use na::allocator::Allocator;
pub use na::dimension::*;
pub use na::storage::Owned;
pub use na::{DefaultAllocator, Dim, DimName};

/// Dual Number structure
///
/// Although `Dual` does implement `PartialEq` and `PartialOrd`,
/// it only compares the real part.
///
/// Additionally, `min` and `max` only compare the real parts, and keep the dual parts.
///
/// Lastly, the `Rem` remainder operator is not correctly or fully defined for `Dual`, and will panic.
#[derive(Clone, Copy)]
pub struct Dual<T: Clone> {
    real: T,
    dual: T,
}
impl<T: Clone> Dual<T> {
    // Implementation of clone method
    pub fn clone(&self) -> Dual<T> {
        Dual {
            real: self.real.clone(),
            dual: self.dual.clone(),
        }
    }
}
impl<T: Clone> Dual<T> {
    /// Create a new dual number from its real and dual parts.
    #[inline]
    pub fn new(real: T, dual: T) -> Dual<T> {
        Dual { real, dual }
    }

    /// Create a new dual number from a real number.
    ///
    /// The dual part is set to zero.
    #[inline]
    pub fn from_real(real: T) -> Dual<T>
    where
        T: Default,
    {
        Dual {
            real,
            dual: T::default(),
        }
    }

    /// Returns the real part
    #[inline]
    pub fn real(&self) -> T {
        self.real.clone()
    }

    /// Returns the real part
    #[inline]
    pub fn real_ref(&self) -> &T {
        &self.real
    }
    /// Returns a mutable reference to the real part
    #[inline]
    pub fn real_mut(&mut self) -> &mut T {
        &mut self.real
    }

    /// Returns the dual part
    #[inline]
    pub fn dual(&self) -> T {
        self.dual.clone()
    }

    /// Returns the dual part
    #[inline]
    pub fn dual_ref(&self) -> &T {
        &self.dual
    }

    /// Returns a mutable reference to the dual part
    #[inline]
    pub fn dual_ref_mut(&mut self) -> &mut T {
        &mut self.dual
    }

    #[inline] //is this necessary?

    pub fn map_dual<F>(&self, r: T, f: F) -> Dual<T>
    where
        F: Fn(&T) -> T,
    {
        let v = f(&self.dual()); //Calls the map method on self to apply the function f to each element of self

        Dual { real: r, dual: v }
    }
}

impl<T: Debug + Clone> Debug for Dual<T> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        f.debug_tuple("Dual")
            .field(&self.real)
            .field(&self.dual)
            .finish()
    }
}
impl<T: Default + Zero + Num + Clone> Zero for Dual<T> {
    #[inline]
    fn zero() -> Dual<T> {
        Dual::from_real(T::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.real() == T::zero()
    }
}
impl<T: Default + Num + Zero + Clone> Default for Dual<T> {
    #[inline]
    fn default() -> Dual<T> {
        Dual::zero()
    }
}
impl<T: Zero + Default + Clone> From<T> for Dual<T> {
    #[inline]
    fn from(realz: T) -> Dual<T> {
        Dual::from_real(realz)
    }
}
impl<T: Clone> Deref for Dual<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.real
    }
}

impl<T: Clone> DerefMut for Dual<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.real
    }
}
/*
impl<T> Dual<T>{
    #[inline]
    pub fn conjugate(self)-> Self{
        self.map(self,|x| x.neg())
    }
}*/
impl<T: Display + Clone> Display for Dual<T> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let precision = f.precision().unwrap_or(4);

        write!(f, "{:.p$}", self.real(), p = precision)?;

        write!(f, " + {:.p$}\u{03B5}{}", self.dual(), p = precision)?;

        Ok(())
    }
}
impl<T: Display + LowerExp + Clone> LowerExp for Dual<T> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let precision = f.precision().unwrap_or(4);

        write!(f, "{:.p$e}{}", self.real(), p = precision)?;
        write!(f, " + {:.p$e}\u{03B5}{}", self.dual(), p = precision)?;

        Ok(())
    }
}
impl<T: PartialEq + Clone> PartialEq<Self> for Dual<T> {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.real() == rhs.real()
    }
}
impl<T: PartialOrd + Clone> PartialOrd<Self> for Dual<T> {
    #[inline]
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(self.real_ref(), rhs.real_ref())
    }

    #[inline]
    fn lt(&self, rhs: &Self) -> bool {
        self.real() < rhs.real()
    }

    #[inline]
    fn gt(&self, rhs: &Self) -> bool {
        self.real() > rhs.real()
    }
}

impl<T: PartialEq + Clone> PartialEq<T> for Dual<T> {
    #[inline]
    fn eq(&self, rhs: &T) -> bool {
        *self.real_ref() == *rhs
    }
}

impl<T: PartialOrd + Clone> PartialOrd<T> for Dual<T> {
    #[inline]
    fn partial_cmp(&self, rhs: &T) -> Option<Ordering> {
        PartialOrd::partial_cmp(self.real_ref(), rhs)
    }

    #[inline]
    fn lt(&self, rhs: &T) -> bool {
        self.real() < *rhs
    }

    #[inline]
    fn gt(&self, rhs: &T) -> bool {
        self.real() > *rhs
    }
}

macro_rules! impl_to_primitive {
    ($($name:ident, $ty:ty),*) => {
        impl<T: Clone+ ToPrimitive> ToPrimitive for Dual<T>
            where

                {
            $(
                #[inline]
                fn $name(&self) -> Option<$ty> {
                    self.real_ref().$name()
                }
            )*
        }
    }
}

macro_rules! impl_from_primitive {
    ($($name:ident, $ty:ty),*) => {
        impl<T:  FromPrimitive> FromPrimitive for Dual<T>
            where
                T: Zero+Clone+Default,

                {
            $(
                #[inline]
                fn $name(n: $ty) -> Option<Dual<T>> {
                    T::$name(n).map(Dual::from_real)
                }
            )*
        }
    }
}

macro_rules! impl_primitive_cast {
    ($($to:ident, $from:ident - $ty:ty),*) => {
        impl_to_primitive!($($to, $ty),*);
        impl_from_primitive!($($from, $ty),*);
    }
}

impl_primitive_cast! {
    to_isize,   from_isize  - isize,
    to_i8,      from_i8     - i8,
    to_i16,     from_i16    - i16,
    to_i32,     from_i32    - i32,
    to_i64,     from_i64    - i64,
    to_usize,   from_usize  - usize,
    to_u8,      from_u8     - u8,
    to_u16,     from_u16    - u16,
    to_u32,     from_u32    - u32,
    to_u64,     from_u64    - u64,
    to_f32,     from_f32    - f32,
    to_f64,     from_f64    - f64
}

impl<T: Num + Clone> Add<T> for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn add(self, rhs: T) -> Dual<T> {
        let mut d = self.clone();
        d.real = d.real + rhs;
        d
    }
}

impl<T: Num + Clone + Default> AddAssign<T> for Dual<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        *self = (*self).clone() + Dual::from_real(rhs)
    }
}

impl<T: Num + Clone> Sub<T> for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn sub(self, rhs: T) -> Dual<T> {
        let mut d = self.clone();
        d.real = d.real - rhs;
        d
    }
}

impl<T: Num + Clone + Default> SubAssign<T> for Dual<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        *self = (*self).clone() - Dual::from_real(rhs)
    }
}

impl<T: Num + Clone + Default> Mul<T> for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn mul(self, rhs: T) -> Dual<T> {
        self * Dual::from_real(rhs)
    }
}

impl<T: Num + Clone + Default> MulAssign<T> for Dual<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = (*self).clone() * Dual::from_real(rhs)
    }
}

impl<T: Num + Clone + Default> Div<T> for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn div(self, rhs: T) -> Dual<T> {
        self / Dual::from_real(rhs)
    }
}

impl<T: Num + Clone + Default> DivAssign<T> for Dual<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = (*self).clone() / Dual::from_real(rhs)
    }
}

impl<T: Signed + Clone> Neg for Dual<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Dual {
            real: self.real.neg(),
            dual: self.dual.neg(),
        }
    }
}

impl<T: Num + Clone> Add for Dual<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Dual {
            real: self.real + rhs.real,
            dual: self.dual + rhs.dual,
        }
    }
}

impl<T: Num + Clone> Add<&Self> for Dual<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: &Self) -> Self {
        Dual {
            real: self.real + rhs.real.clone(),
            dual: self.dual + rhs.dual.clone(),
        }
    }
}

impl<T: Num + Clone> AddAssign<Self> for Dual<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        let tmp = (*self).clone();
        *self = tmp + rhs;

        //*self = (*self) + rhs
    }
}

impl<T: Num + Clone> Sub<Self> for Dual<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Dual {
            real: self.real - rhs.real,
            dual: self.dual - rhs.dual,
        }
    }
}

impl<T: Num + Clone> Sub<&Self> for Dual<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self {
        Dual {
            real: self.real - rhs.real.clone(),
            dual: self.dual - rhs.dual.clone(),
        }
    }
}

impl<T: Num + Clone> SubAssign<Self> for Dual<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = (*self).clone() - rhs
    }
}

impl<T: Num + Clone> Mul<&Self> for Dual<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self {
        // TODO: skip real part

        // let real = self.real * rhs.real;
        let v = rhs.real.clone() * self.dual.clone() + self.real.clone() * rhs.dual.clone();

        Dual {
            real: self.real.clone() * rhs.real.clone(),
            dual: v,
        }
    }
}

impl<T: Num + Clone> Mul<Self> for Dual<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * (&rhs)
    }
}

impl<T: Num + Clone> MulAssign<Self> for Dual<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = (*self).clone() * rhs
    }
}

macro_rules! impl_mul_add {
    ($(<$a:ident, $b:ident>),*) => {
        $(
            impl<T:  Num+Clone +Default+ Mul + Add> MulAdd<$a, $b> for Dual<T> {
                type Output = Dual<T>;

                #[inline]
                fn mul_add(self, a: $a, b: $b) -> Dual<T> {
                    (self * a) + b
                }
            }

            impl<T:  Clone+Num +Default+ Mul + Add> MulAddAssign<$a, $b> for Dual<T> {
                #[inline]
                fn mul_add_assign(&mut self, a: $a, b: $b) {
                    *self = ((*self).clone() * a) + b;
                }
            }
        )*
    }
}

impl_mul_add! {
    <Self, Self>,
    <T, Self>,
    <Self, T>,
    <T, T>
}

impl<T: Num + Clone> Div<&Self> for Dual<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self {
        // TODO: specialize with inv so we can precompute the inverse
        let d = rhs.real.clone() * rhs.real.clone();

        /*
            _dual = (_dual * x.rpart() - _real * x.dpart()) / (x.rpart() * x.rpart());
        _real = _real / x.rpart();
            */
        let vdpart =
            (self.dual.clone() * rhs.real.clone() - self.real.clone() * rhs.dual.clone()) / d;
        let vrpart = self.real.clone() / rhs.real.clone();
        Dual {
            real: vrpart,
            dual: vdpart,
        }
    }
}

impl<T: Num + Clone> Div<Self> for Dual<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        self / (&rhs)
    }
}

impl<T: Num + Clone> DivAssign<Self> for Dual<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = (*self).clone() / rhs
    }
}

impl<T: Num + Clone> Rem<Self> for Dual<T> {
    type Output = Self;

    /// **UNIMPLEMENTED!!!**
    ///
    /// As far as I know, remainder is not a valid operation on dual numbers,
    /// but is required for the `Float` trait to be implemented.
    fn rem(self, _: Self) -> Self {
        unimplemented!()
    }
}

impl<T: Num + Clone> Rem<&Self> for Dual<T> {
    type Output = Self;

    /// **UNIMPLEMENTED!!!**
    ///
    /// As far as I know, remainder is not a valid operation on dual numbers,
    /// but is required for the `Float` trait to be implemented.
    fn rem(self, _: &Self) -> Self {
        unimplemented!()
    }
}

impl<T: Num + Clone> RemAssign<Self> for Dual<T> {
    /// **UNIMPLEMENTED!!!**
    ///
    /// As far as I know, remainder is not a valid operation on dual numbers,
    /// but is required for the `Float` trait to be implemented.
    fn rem_assign(&mut self, _: Self) {
        unimplemented!()
    }
}

impl<T: Clone, P: Into<Dual<T>>> Pow<P> for Dual<T>
where
    Dual<T>: Float,
{
    type Output = Dual<T>;

    #[inline]
    fn pow(self, rhs: P) -> Dual<T> {
        self.powf(rhs.into())
    }
}

impl<T: Clone> Inv for Dual<T>
where
    Self: One + Div<Output = Self>,
{
    type Output = Dual<T>;

    #[inline]
    fn inv(self) -> Dual<T> {
        Dual::one() / self
    }
}

impl<T: Default + Clone> Signed for Dual<T>
where
    T: Signed + PartialOrd,
{
    #[inline]
    fn abs(&self) -> Self {
        let s = self.real().clone().signum();
        Dual {
            real: self.real.clone() * s.clone(),
            dual: self.dual.clone() * s,
        }
    }

    #[inline]
    fn abs_sub(&self, rhs: &Self) -> Self {
        if self.real() > rhs.real() {
            (self).clone().sub((*rhs).clone())
        } else {
            Self::zero()
        }
    }

    #[inline]
    fn signum(&self) -> Self {
        Dual::from_real(self.real().signum())
    }

    #[inline]
    fn is_positive(&self) -> bool {
        self.real().is_positive()
    }

    #[inline]
    fn is_negative(&self) -> bool {
        self.real().is_negative()
    }
}

impl<T: Unsigned + Clone> Unsigned for Dual<T> where Self: Num {}

impl<T: Num + One + Clone + Default> One for Dual<T> {
    #[inline]
    fn one() -> Dual<T> {
        Dual::from_real(T::one())
    }

    #[inline]
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        self.real().is_one()
    }
}

impl<T: Num + Default + Clone> Num for Dual<T> {
    type FromStrRadixErr = <T as Num>::FromStrRadixErr;

    #[inline]
    fn from_str_radix(str: &str, radix: u32) -> Result<Dual<T>, Self::FromStrRadixErr> {
        <T as Num>::from_str_radix(str, radix).map(Dual::from_real)
    }
}

impl<T: Float + Default> NumCast for Dual<T> {
    #[inline]
    fn from<P: ToPrimitive>(n: P) -> Option<Dual<T>> {
        <T as NumCast>::from(n).map(Dual::from_real)
    }
}

macro_rules! impl_float_const {
    ($($c:ident),*) => {
        $(
            fn $c() -> Dual<T> { Dual::from_real(T::$c()) }
        )*
    }
}

impl<T: FloatConst + Zero + Clone + Default> FloatConst for Dual<T> {
    impl_float_const!(
        E,
        FRAC_1_PI,
        FRAC_1_SQRT_2,
        FRAC_2_PI,
        FRAC_2_SQRT_PI,
        FRAC_PI_2,
        FRAC_PI_3,
        FRAC_PI_4,
        FRAC_PI_6,
        FRAC_PI_8,
        LN_10,
        LN_2,
        LOG10_E,
        LOG2_E,
        PI,
        SQRT_2
    );
}

macro_rules! impl_real_constant {
    ($($prop:ident),*) => {
        $(
            fn $prop() -> Self { Dual::from_real(<T as Float>::$prop()) }
        )*
    }
}

macro_rules! impl_single_boolean_op {
    ($op:ident REAL) => {
        fn $op(self) -> bool {
            self.real().$op()
        }
    };
    ($op:ident OR) => {
        fn $op(self) -> bool {
            let mut b = self.real().$op();

            b |= self.dual().$op();

            b
        }
    };
    ($op:ident AND) => {
        fn $op(self) -> bool {
            let mut b = self.real().$op();

            b &= self.dual().$op();

            b
        }
    };
}

macro_rules! impl_boolean_op {
    ($($op:ident $t:tt),*) => {
        $(impl_single_boolean_op!($op $t);)*
    };
}

macro_rules! impl_real_op {
    ($($op:ident),*) => {
        $(
            fn $op(self) -> Self { Dual::from_real(self.real().$op()) }
        )*
    }
}

impl<T: Num + Zero + Clone + Default> Sum for Dual<T> {
    fn sum<I: Iterator<Item = Dual<T>>>(iter: I) -> Dual<T> {
        iter.fold(Dual::zero(), |a, b| a + b)
    }
}

impl<'a, T: Num + Zero + Clone + Default> Sum<&'a Dual<T>> for Dual<T> {
    fn sum<I: Iterator<Item = &'a Dual<T>>>(iter: I) -> Dual<T> {
        iter.fold(Dual::zero(), |a, b| a + (*b).clone())
    }
}

impl<T: Clone + Num + One + Default> Product for Dual<T> {
    fn product<I: Iterator<Item = Dual<T>>>(iter: I) -> Dual<T> {
        iter.fold(Dual::one(), |a, b| a * b)
    }
}

impl<'a, T: Num + One + Clone + Default> Product<&'a Dual<T>> for Dual<T> {
    fn product<I: Iterator<Item = &'a Dual<T>>>(iter: I) -> Dual<T> {
        iter.fold(Dual::one(), |a, b| a * (*b).clone())
    }
}

impl<T: Default> Float for Dual<T>
where
    T: Float + Signed + FloatConst + Clone,
{
    impl_real_constant!(
        nan,
        infinity,
        neg_infinity,
        neg_zero,
        min_positive_value,
        epsilon,
        min_value,
        max_value
    );

    impl_boolean_op!(
        is_nan              OR,
        is_infinite         OR,
        is_finite           AND,
        is_normal           AND,
        is_sign_positive    REAL,
        is_sign_negative    REAL
    );

    #[inline]
    fn classify(self) -> FpCategory {
        self.real().classify()
    }

    impl_real_op!(floor, ceil, round, trunc);

    #[inline]
    fn fract(self) -> Self {
        let v = self.clone();

        Dual {
            real: self.real().fract(),
            dual: v.dual(),
        }
    }

    #[inline]
    fn signum(self) -> Self {
        Dual::from_real(self.real().signum())
    }

    #[inline]
    fn abs(self) -> Self {
        let s = self.real().signum();

        Dual {
            real: self.real * s,
            dual: self.dual * s,
        }
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        if self.real() > other.real() {
            self
        } else {
            other
        }
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        if self.real() < other.real() {
            self
        } else {
            other
        }
    }

    #[inline]
    fn abs_sub(self, rhs: Self) -> Self {
        if self.real() > rhs.real() {
            self.sub(rhs)
        } else {
            Self::zero()
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let dualx = Dual::from_real(self.real().mul_add(a.real(), b.real()));

        //check this!!!
        let dualy = self.dual() * a.real() + self.real() * a.dual() + b.dual();

        Dual {
            real: dualx.real,
            dual: dualy,
        }
    }

    #[inline]
    fn recip(self) -> Self {
        Self::one() / self
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        let r = self.real().powi(n - 1);
        let nf = <T as NumCast>::from(n).expect("Invalid value") * r;

        self.map_dual(self.real() * r, |x| nf * *x)
    }

    #[inline]
    /*
       template<class T, class U, CPPDUALS_ENABLE_SAME_DEPTH_AND_COMMON_T(T,U)>
    common_t
    pow(const dual<T> & f, const dual<U> & g) {
      using std::pow;
      using std::log;
      T v = pow(f.rpart(), g.rpart());
      return common_t(v,
                      pow(f.rpart(), g.rpart() - T(1)) *
                      (g.rpart() * f.dpart()
                       + f.rpart() * log(f.rpart()) * g.dpart()));
    }

    template<class T, class U, CPPDUALS_ENABLE_LEQ_DEPTH_AND_COMMON_T(T,U)>
    common_t
    pow(const dual<T> & x, const U & y) {
      using std::pow;
      return common_t(pow(x.rpart(), y),
                      x.dpart() * y * pow(x.rpart(), y - U(1)));
    }

    fn powf(self, n: Self) -> Self {
            let a = self.real().powf(n.real());
            let a = n.real() * self.real().powf(n.real() - T::one());
            let b = c * self.real().ln();

            let mut v = self.zip_map(&n, |x, y| a * x + b * y);

            Dual{real: c, dual: v.dual()}
        }
        check this!!
        */

    fn powf(self, n: Self) -> Self {
        let c = self.real.powf(n.real);
        let a = n.real * (self.real.powf(n.real - T::one()));
        let b = c * self.real.ln();

        let dualp = a * self.dual + b * n.dual;

        Dual {
            real: c,
            dual: dualp,
        }
    }

    #[inline]
    fn exp(self) -> Self {
        let real = self.real().exp();
        self.map_dual(real, |x| real * *x)
    }

    #[inline]
    fn exp2(self) -> Self {
        let real = self.real().exp2();
        self.map_dual(real, |x| *x * T::LN_2() * real)
    }

    #[inline]
    fn ln(self) -> Self {
        self.map_dual(self.real().ln(), |x| *x / self.real())
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[inline]
    fn log2(self) -> Self {
        self.map_dual(self.real().log2(), |x| *x / (self.real() * T::LN_2()))
    }

    #[inline]
    fn log10(self) -> Self {
        self.map_dual(self.real().log10(), |x| *x / (self.real() * T::LN_10()))
    }

    #[inline]
    fn sqrt(self) -> Self {
        let real = self.real().sqrt();
        let d = T::from(1).unwrap() / (T::from(2).unwrap() * real);
        self.map_dual(real, |x| *x * d)
    }

    #[inline]
    fn cbrt(self) -> Self {
        let real = self.real().cbrt();
        self.map_dual(real, |x| *x / (T::from(3).unwrap() * real))
    }

    #[inline]

    fn hypot(self, other: Self) -> Self {
        let c = self.real().hypot(other.real());
        let dualx = (self.real() * other.dual() + other.real() * self.dual()) / c;

        Dual {
            real: c,
            dual: dualx,
        }
    }

    #[inline]
    fn sin(self) -> Self {
        let c = self.real().cos();
        self.map_dual(self.real().sin(), |x| *x * c)
    }

    #[inline]
    fn cos(self) -> Self {
        let c = self.real().sin();
        self.map_dual(self.real().cos(), |x| x.neg() * c)
    }

    #[inline]
    fn tan(self) -> Self {
        let t = self.real().tan();
        let c = t * t + T::one();
        self.map_dual(t, |x| *x * c)
    }

    #[inline]
    fn asin(self) -> Self {
        // TODO: implement inv
        let c = (T::one() - self.real().powi(2)).sqrt();
        self.map_dual(self.real().asin(), |x| *x / c)
    }

    #[inline]
    fn acos(self) -> Self {
        // TODO: implement inv
        let c = (T::one() - self.real().powi(2)).sqrt();
        self.map_dual(self.real().acos(), |x| x.neg() / c)
    }

    #[inline]
    fn atan(self) -> Self {
        // TODO: implement inv
        let c = (self.real().powi(2) + T::one()).sqrt();
        self.map_dual(self.real().atan(), |x| *x / c)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        let c = self.real.powi(2) + other.real.powi(2);
        let dualx = (other.real() * self.dual() - self.real() * other.dual()) / c;

        Dual {
            real: self.real().atan2(other.real()),
            dual: dualx,
        }
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.real().sin_cos();
        let sn = self.map_dual(s, |x| *x * c);
        let cn = self.map_dual(c, |x| x.neg() * s);
        (sn, cn)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        let c = self.real().exp();
        self.map_dual(self.real().exp_m1(), |x| *x * c)
    }

    #[inline]
    fn ln_1p(self) -> Self {
        let c = self.real() + T::one();
        self.map_dual(self.real().ln_1p(), |x| *x / c)
    }

    #[inline]
    fn sinh(self) -> Self {
        let c = self.real().cosh();
        self.map_dual(self.real().sinh(), |x| *x * c)
    }

    #[inline]
    fn cosh(self) -> Self {
        let c = self.real().sinh();
        self.map_dual(self.real().cosh(), |x| *x * c)
    }

    #[inline]
    fn tanh(self) -> Self {
        let real = self.real().tanh();
        let c = T::one() - real.powi(2);
        self.map_dual(real, |x| *x * c)
    }

    #[inline]
    fn asinh(self) -> Self {
        let c = (self.real().powi(2) + T::one()).sqrt();
        self.map_dual(self.real().asinh(), |x| *x / c)
    }

    #[inline]
    fn acosh(self) -> Self {
        let c = (self.real() + T::one()).sqrt() * (self.real() - T::one()).sqrt();
        self.map_dual(self.real().acosh(), |x| *x / c)
    }

    #[inline]
    fn atanh(self) -> Self {
        let c = T::one() - self.real().powi(2);
        self.map_dual(self.real().atanh(), |x| *x / c)
    }

    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.real().integer_decode()
    }

    #[inline]
    fn to_degrees(self) -> Self {
        Dual::from_real(self.real().to_degrees())
    }

    #[inline]
    fn to_radians(self) -> Self {
        Dual::from_real(self.real().to_radians())
    }
}

// TODO
// impl<T: na::Real> na::Real for Dual<T> {}
