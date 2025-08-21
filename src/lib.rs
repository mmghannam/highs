#![forbid(missing_docs)]
//! Safe rust binding to the [HiGHS](https://highs.dev) linear programming solver.
//!
//! ## Usage example
//!
//! ### Building a problem constraint by constraint with [RowProblem]
//!
//! Useful for traditional problem modelling where you first declare your variables, then add
//!constraints one by one.
//!
//! ```
//! use highs::{Sense, Model, HighsModelStatus, RowProblem, LikeModel};
//! // max: x + 2y + z
//! // under constraints:
//! // c1: 3x +  y      <= 6
//! // c2:       y + 2z <= 7
//! let mut pb = RowProblem::default();
//! // Create a variable named x, with a coefficient of 1 in the objective function,
//! // that is bound between 0 and +∞.
//! let x = pb.add_column(1., 0..);
//! let y = pb.add_column(2., 0..);
//! let z = pb.add_column(1., 0..);
//! // constraint c1: x*3 + y*1 is bound to ]-∞; 6]
//! pb.add_row(..=6, &[(x, 3.), (y, 1.)]);
//! // constraint c2: y*1 +  z*2 is bound to ]-∞; 7]
//! pb.add_row(..=7, &[(y, 1.), (z, 2.)]);
//!
//! let solved = pb.optimise(Sense::Maximise).solve();
//!
//! assert_eq!(solved.status(), HighsModelStatus::Optimal);
//!
//! let solution = solved.get_solution();
//! // The expected solution is x=0  y=6  z=0.5
//! assert_eq!(solution.columns(), vec![0., 6., 0.5]);
//! // All the constraints are at their maximum
//! assert_eq!(solution.rows(), vec![6., 7.]);
//! ```
//!
//! ### Building a problem variable by variable with [ColProblem]
//!
//! Useful for resource allocation problems and other problems when you know in advance the number
//! of constraints and their bounds, but dynamically add new variables to the problem.
//!
//! This is slightly more efficient than building the problem constraint by constraint.
//!
//! ```
//! use highs::{ColProblem, Sense};
//! let mut pb = ColProblem::new();
//! // We cannot use more then 5 units of sugar in total.
//! let sugar = pb.add_row(..=5);
//! // We cannot use more then 3 units of milk in total.
//! let milk = pb.add_row(..=3);
//! // We have a first cake that we can sell for 2€. Baking it requires 1 unit of milk and 2 of sugar.
//! pb.add_integer_column(2., 0.., &[(sugar, 2.), (milk, 1.)]);
//! // We have a second cake that we can sell for 8€. Baking it requires 2 units of milk and 3 of sugar.
//! pb.add_integer_column(8., 0.., &[(sugar, 3.), (milk, 2.)]);
//! // Find the maximal possible profit
//! let solution = pb.optimise(Sense::Maximise).solve().get_solution();
//! // The solution is to bake 1 cake of each sort
//! assert_eq!(solution.columns(), vec![1., 1.]);
//! ```
//!
//! ```
//! use highs::{Sense, Model, HighsModelStatus, ColProblem, LikeModel};
//! // max: x + 2y + z
//! // under constraints:
//! // c1: 3x +  y      <= 6
//! // c2:       y + 2z <= 7
//! let mut pb = ColProblem::default();
//! let c1 = pb.add_row(..6.);
//! let c2 = pb.add_row(..7.);
//! // x
//! pb.add_column(1., 0.., &[(c1, 3.)]);
//! // y
//! pb.add_column(2., 0.., &[(c1, 1.), (c2, 1.)]);
//! // z
//! pb.add_column(1., 0.., vec![(c2, 2.)]);
//!
//! let solved = pb.optimise(Sense::Maximise).solve();
//!
//! assert_eq!(solved.status(), HighsModelStatus::Optimal);
//!
//! let solution = solved.get_solution();
//! // The expected solution is x=0  y=6  z=0.5
//! assert_eq!(solution.columns(), vec![0., 6., 0.5]);
//! // All the constraints are at their maximum
//! assert_eq!(solution.rows(), vec![6., 7.]);
//! ```
//!
//! ### Integer variables
//!
//! HiGHS supports mixed integer-linear programming.
//! You can use `add_integer_column` to add an integer variable to the problem,
//! and the solution is then guaranteed to contain a whole number as a value for this variable.
//!
//! ```
//! use highs::{Sense, Model, HighsModelStatus, ColProblem};
//! // maximize: x + 2y under constraints x + y <= 3.5 and x - y >= 1
//! let mut pb = ColProblem::default();
//! let c1 = pb.add_row(..3.5);
//! let c2 = pb.add_row(1..);
//! // x (continuous variable)
//! pb.add_column(1., 0.., &[(c1, 1.), (c2, 1.)]);
//! // y (integer variable)
//! pb.add_integer_column(2., 0.., &[(c1, 1.), (c2, -1.)]);
//! let solved = pb.optimise(Sense::Maximise).solve();
//! // The expected solution is x=2.5  y=1
//! assert_eq!(solved.get_solution().columns(), vec![2.5, 1.]);
//! ```

use highs_sys::*;
use std::convert::{TryFrom, TryInto};
use std::ffi::{c_void, CString};
use std::num::TryFromIntError;
use std::ops::{Bound, Index, RangeBounds};
use std::os::raw::c_int;
use std::ptr::null_mut;

pub use matrix_col::{ColMatrix, Row};
pub use matrix_row::{Col, RowMatrix};
pub use status::{HighsModelStatus, HighsStatus};

use crate::options::HighsOptionValue;

/// A problem where variables are declared first, and constraints are then added dynamically.
/// See [`Problem<RowMatrix>`](Problem#impl-1).
pub type RowProblem = Problem<RowMatrix>;
/// A problem where constraints are declared first, and variables are then added dynamically.
/// See [`Problem<ColMatrix>`](Problem#impl).
pub type ColProblem = Problem<ColMatrix>;

mod matrix_col;
mod matrix_row;
mod options;
mod status;

/// A complete optimization problem.
/// Depending on the `MATRIX` type parameter, the problem will be built
/// constraint by constraint (with [ColProblem]), or
/// variable by variable (with [RowProblem])
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Problem<MATRIX = ColMatrix> {
    // columns
    colcost: Vec<f64>,
    collower: Vec<f64>,
    colupper: Vec<f64>,
    // rows
    rowlower: Vec<f64>,
    rowupper: Vec<f64>,
    integrality: Option<Vec<HighsInt>>,
    matrix: MATRIX,
}

impl<MATRIX: Default> Problem<MATRIX>
where
    Problem<ColMatrix>: From<Problem<MATRIX>>,
{
    /// Number of variables in the problem
    pub fn num_cols(&self) -> usize {
        self.colcost.len()
    }

    /// Number of constraints in the problem
    pub fn num_rows(&self) -> usize {
        self.rowlower.len()
    }

    fn add_row_inner<N: Into<f64> + Copy, B: RangeBounds<N>>(&mut self, bounds: B) -> Row {
        let r = self.num_rows().try_into().expect("too many rows");
        let low = bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        let high = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
        self.rowlower.push(low);
        self.rowupper.push(high);
        r
    }

    fn add_column_inner<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        col_factor: f64,
        bounds: B,
        is_integral: bool,
    ) {
        if is_integral && self.integrality.is_none() {
            self.integrality = Some(vec![0; self.num_cols()]);
        }
        if let Some(integrality) = &mut self.integrality {
            integrality.push(if is_integral { 1 } else { 0 });
        }
        self.colcost.push(col_factor);
        let low = bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        let high = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
        self.collower.push(low);
        self.colupper.push(high);
    }

    /// Create a model based on this problem. Don't solve it yet.
    /// If the problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    /// If the problem is invalid (according to HiGHS), this function will panic.
    pub fn optimise(self, sense: Sense) -> Model {
        self.try_optimise(sense).expect("invalid problem")
    }

    /// Create a model based on this problem. Don't solve it yet.
    /// If the problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    pub fn try_optimise(self, sense: Sense) -> Result<Model, HighsStatus> {
        let mut m = Model::try_new(self)?;
        m.set_sense(sense);
        Ok(m)
    }

    /// Create a new problem instance
    pub fn new() -> Self {
        Self::default()
    }
}

fn bound_value<N: Into<f64> + Copy>(b: Bound<&N>) -> Option<f64> {
    match b {
        Bound::Included(v) | Bound::Excluded(v) => Some((*v).into()),
        Bound::Unbounded => None,
    }
}

fn c(n: usize) -> HighsInt {
    n.try_into().expect("size too large for HiGHS")
}

macro_rules! highs_call {
    ($function_name:ident ($($param:expr),+)) => {
        try_handle_status(
            $function_name($($param),+),
            stringify!($function_name)
        )
    }
}

/// A model to solve
#[derive(Debug, Clone)]
pub struct Model {
    highs: HighsPtr,
}

/// A solved model
#[derive(Debug, Clone)]
pub struct SolvedModel {
    highs: HighsPtr,
}

/// Whether to maximize or minimize the objective function
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Sense {
    /// max
    Maximise = OBJECTIVE_SENSE_MAXIMIZE as isize,
    /// min
    Minimise = OBJECTIVE_SENSE_MINIMIZE as isize,
}

impl From<HighsInt> for Sense {
    fn from(value: HighsInt) -> Self {
        match value {
            OBJECTIVE_SENSE_MINIMIZE => Sense::Minimise,
            OBJECTIVE_SENSE_MAXIMIZE => Sense::Maximise,
            other => {
                panic!("Unknown objective sense with value {}", other)
            }
        }
    }
}

/// Type of a variable in the problem
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum VarType {
    /// Continuous variable (real valued)
    Continuous,
    /// Integer only variable
    Integer,
    /// Implicit Integer
    ImplicitInteger,
    /// Semi-Integer
    SemiInteger,
}

#[allow(non_upper_case_globals)]
impl From<HighsInt> for VarType {
    fn from(value: HighsInt) -> Self {
        match value {
            kHighsVarTypeContinuous => VarType::Continuous,
            kHighsVarTypeInteger => VarType::Integer,
            kHighsVarTypeImplicitInteger => VarType::ImplicitInteger,
            kHighsVarTypeSemiInteger => VarType::SemiInteger,
            other => {
                panic!("Unknown variable type with value {}", other)
            }
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new::<Problem<ColMatrix>>(Problem::default())
    }
}

impl Model {
    /// number of columns
    pub fn num_cols(&self) -> usize {
        unsafe { Highs_getNumCols(self.highs.ptr()) as usize }
    }

    /// number of rows
    pub fn num_rows(&self) -> usize {
        unsafe { Highs_getNumRows(self.highs.ptr()) as usize }
    }

    /// number of non-zeros
    pub fn num_nz(&self) -> usize {
        unsafe { Highs_getNumNz(self.highs.ptr()) as usize }
    }

    /// Returns the LP data in the problem
    pub fn get_row_lp(
        &self,
    ) -> (
        usize,
        usize,
        usize,
        Sense,
        f64,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<Vec<(Col, f64)>>,
        Vec<VarType>,
    ) {
        let mut num_col = unsafe { Highs_getNumCol(self.highs.ptr()) };
        let mut num_row = unsafe { Highs_getNumRow(self.highs.ptr()) };
        let mut num_nz = unsafe { Highs_getNumNz(self.highs.ptr()) };

        // Allocate arrays with the appropriate sizes
        let mut col_cost = vec![0.0; num_col as usize];
        let mut col_lower = vec![0.0; num_col as usize];
        let mut col_upper = vec![0.0; num_col as usize];
        let mut row_lower = vec![0.0; num_row as usize];
        let mut row_upper = vec![0.0; num_row as usize];
        let mut a_start = vec![0; num_row as usize];
        let mut a_index = vec![0; num_nz as usize];
        let mut a_value = vec![0.0; num_nz as usize];
        let mut integrality = vec![0; num_col as usize];

        let mut sense = 0i32;
        let mut offset: f64 = 0.0;

        let res = unsafe {
            highs_call! {
                Highs_getLp(
                    self.highs.ptr(),
                    kHighsMatrixFormatRowwise,
                    &mut num_col,
                    &mut num_row,
                    &mut num_nz,
                    &mut sense,
                    &mut offset,
                    col_cost.as_mut_ptr(),
                    col_lower.as_mut_ptr(),
                    col_upper.as_mut_ptr(),
                    row_lower.as_mut_ptr(),
                    row_upper.as_mut_ptr(),
                    a_start.as_mut_ptr(),
                    a_index.as_mut_ptr(),
                    a_value.as_mut_ptr(),
                    integrality.as_mut_ptr()
                )
            }
        };

        let num_col = num_col as usize;
        let num_row = num_row as usize;
        let num_nz = num_nz as usize;

        let integrality = integrality.iter().map(|v| (*v).into()).collect();

        if res.is_err() {
            panic!(
                "Failed to call Highs_getLp, got error {:?}",
                res.unwrap_err()
            );
        }

        let mut row_data = Vec::with_capacity(num_row);
        // println!("num_row: {}", num_row);
        // println!("num_col: {}", num_col);
        // println!("a_start: {:?}", a_start);
        // println!("a_index: {:?}", a_index);
        // println!("a_value: {:?}", a_value);
        for i in 0..num_row {
            let start = a_start[i] as usize;
            let end = if i == (num_row - 1) {
                num_nz
            } else {
                a_start[i + 1] as usize
            };

            let mut index_vals = Vec::with_capacity(end - start);
            for j in start..end {
                index_vals.push((a_index[j] as Col, a_value[j]));
            }
            row_data.push(index_vals);
        }

        (
            num_col,
            num_row,
            num_nz,
            sense.into(), // Convert to your Sense enum
            offset,
            col_cost.clone(),
            col_lower.clone(),
            col_upper.clone(),
            row_lower.clone(),
            row_upper.clone(),
            row_data.clone(),
            integrality,
        )
    }

    /// Changes the integrality of a column in the model.
    pub fn change_col_integrality(&mut self, col: Col, integer: bool) -> Result<(), HighsStatus> {
        unsafe {
            highs_call! {
                Highs_changeColIntegrality(
                    self.highs.mut_ptr(),
                    col as c_int,
                    integer as c_int
                )
            }?
        };

        Ok(())
    }

    /// Returns the LP data in the presolved problem
    pub fn get_presolved_row_lp(
        &self,
    ) -> (
        usize,
        usize,
        usize,
        Sense,
        f64,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<Vec<(Col, f64)>>,
        Vec<VarType>,
    ) {
        let mut num_col = unsafe { Highs_getPresolvedNumCol(self.highs.ptr()) };
        let mut num_row = unsafe { Highs_getPresolvedNumRow(self.highs.ptr()) };
        let mut num_nz = unsafe { Highs_getPresolvedNumNz(self.highs.ptr()) };

        // Allocate arrays with the appropriate sizes
        let mut col_cost = vec![0.0; num_col as usize];
        let mut col_lower = vec![0.0; num_col as usize];
        let mut col_upper = vec![0.0; num_col as usize];
        let mut row_lower = vec![0.0; num_row as usize];
        let mut row_upper = vec![0.0; num_row as usize];
        let mut a_start = vec![0; num_row as usize];
        let mut a_index = vec![0; num_nz as usize];
        let mut a_value = vec![0.0; num_nz as usize];
        let mut integrality = vec![0; num_col as usize];

        let mut sense = 0i32;
        let mut offset: f64 = 0.0;

        let res = unsafe {
            highs_call! {
                Highs_getPresolvedLp(
                    self.highs.ptr(),
                    kHighsMatrixFormatRowwise,
                    &mut num_col,
                    &mut num_row,
                    &mut num_nz,
                    &mut sense,
                    &mut offset,
                    col_cost.as_mut_ptr(),
                    col_lower.as_mut_ptr(),
                    col_upper.as_mut_ptr(),
                    row_lower.as_mut_ptr(),
                    row_upper.as_mut_ptr(),
                    a_start.as_mut_ptr(),
                    a_index.as_mut_ptr(),
                    a_value.as_mut_ptr(),
                    integrality.as_mut_ptr()
                )
            }
        };

        let num_col = num_col as usize;
        let num_row = num_row as usize;
        let num_nz = num_nz as usize;

        let integrality = integrality.iter().map(|v| (*v).into()).collect();

        if res.is_err() {
            panic!(
                "Failed to call Highs_getLp, got error {:?}",
                res.unwrap_err()
            );
        }

        let mut row_data = Vec::with_capacity(num_row);
        // println!("num_row: {}", num_row);
        // println!("num_col: {}", num_col);
        // println!("a_start: {:?}", a_start);
        // println!("a_index: {:?}", a_index);
        // println!("a_value: {:?}", a_value);
        for i in 0..num_row {
            let start = a_start[i] as usize;
            let end = if i == (num_row - 1) {
                num_nz
            } else {
                a_start[i + 1] as usize
            };

            let mut index_vals = Vec::with_capacity(end - start);
            for j in start..end {
                index_vals.push((a_index[j] as Col, a_value[j]));
            }
            row_data.push(index_vals);
        }

        (
            num_col,
            num_row,
            num_nz,
            sense.into(), // Convert to your Sense enum
            offset,
            col_cost.clone(),
            col_lower.clone(),
            col_upper.clone(),
            row_lower.clone(),
            row_upper.clone(),
            row_data.clone(),
            integrality,
        )
    }
    /// Presolve the current model
    pub fn presolve(&mut self) {
        let ret = unsafe { Highs_presolve(self.highs.mut_ptr()) };
        assert_eq!(ret, STATUS_OK, "runPresolve failed");
    }

    /// Set the optimization sense (minimize by default)
    pub fn set_sense(&mut self, sense: Sense) {
        let ret = unsafe { Highs_changeObjectiveSense(self.highs.mut_ptr(), sense as c_int) };
        assert_eq!(ret, STATUS_OK, "changeObjectiveSense failed");
    }

    /// Reads a problem
    pub fn read(&mut self, path: &str) {
        let c_path = CString::new(path).expect("invalid path");
        unsafe {
            highs_call!(Highs_readModel(self.highs.mut_ptr(), c_path.as_ptr()))
                .expect("failed to read model");
        }
    }

    /// Create a Highs model to be optimized (but don't solve it yet).
    /// If the given problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    /// Panics if the problem is incoherent
    pub fn new<P: Into<Problem<ColMatrix>>>(problem: P) -> Self {
        Self::try_new(problem).expect("incoherent problem")
    }

    /// Create a Highs model to be optimized (but don't solve it yet).
    /// If the given problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    /// Returns an error if the problem is incoherent
    pub fn try_new<P: Into<Problem<ColMatrix>>>(problem: P) -> Result<Self, HighsStatus> {
        let mut highs = HighsPtr::default();
        highs.make_quiet();
        let problem = problem.into();
        log::debug!(
            "Adding a problem with {} variables and {} constraints to HiGHS",
            problem.num_cols(),
            problem.num_rows()
        );
        let offset = 0.0;
        unsafe {
            if let Some(integrality) = &problem.integrality {
                highs_call!(Highs_passMip(
                    highs.mut_ptr(),
                    c(problem.num_cols()),
                    c(problem.num_rows()),
                    c(problem.matrix.avalue.len()),
                    MATRIX_FORMAT_COLUMN_WISE,
                    OBJECTIVE_SENSE_MINIMIZE,
                    offset,
                    problem.colcost.as_ptr(),
                    problem.collower.as_ptr(),
                    problem.colupper.as_ptr(),
                    problem.rowlower.as_ptr(),
                    problem.rowupper.as_ptr(),
                    problem.matrix.astart.as_ptr(),
                    problem.matrix.aindex.as_ptr(),
                    problem.matrix.avalue.as_ptr(),
                    integrality.as_ptr()
                ))
            } else {
                highs_call!(Highs_passLp(
                    highs.mut_ptr(),
                    c(problem.num_cols()),
                    c(problem.num_rows()),
                    c(problem.matrix.avalue.len()),
                    MATRIX_FORMAT_COLUMN_WISE,
                    OBJECTIVE_SENSE_MINIMIZE,
                    offset,
                    problem.colcost.as_ptr(),
                    problem.collower.as_ptr(),
                    problem.colupper.as_ptr(),
                    problem.rowlower.as_ptr(),
                    problem.rowupper.as_ptr(),
                    problem.matrix.astart.as_ptr(),
                    problem.matrix.aindex.as_ptr(),
                    problem.matrix.avalue.as_ptr()
                ))
            }
            .map(|_| Self { highs })
        }
    }

    /// Prevents writing anything to the standard output or to files when solving the model
    pub fn make_quiet(&mut self) {
        self.highs.make_quiet()
    }

    /// Set a custom parameter on the model.
    /// For the list of available options and their documentation, see:
    /// <https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.html>
    ///
    /// ```
    /// # use highs::ColProblem;
    /// # use highs::Sense::Maximise;
    /// let mut model = ColProblem::default().optimise(Maximise);
    /// model.set_option("presolve", "off"); // disable the presolver
    /// model.set_option("solver", "ipm"); // use the ipm solver
    /// model.set_option("time_limit", 30.0); // stop after 30 seconds
    /// model.set_option("parallel", "on"); // use multiple cores
    /// model.set_option("threads", 4); // solve on 4 threads
    /// ```
    pub fn set_option<STR: Into<Vec<u8>>, V: HighsOptionValue>(&mut self, option: STR, value: V) {
        self.highs.set_option(option, value)
    }

    /// Find the optimal value for the problem, panic if the problem is incoherent
    pub fn solve(self) -> SolvedModel {
        self.try_solve().expect("HiGHS error: invalid problem")
    }

    /// Find the optimal value for the problem, return an error if the problem is incoherent
    pub fn try_solve(mut self) -> Result<SolvedModel, HighsStatus> {
        unsafe { highs_call!(Highs_run(self.highs.mut_ptr())) }
            .map(|_| SolvedModel { highs: self.highs })
    }

    /// Adds a new constraint to the highs model.
    ///
    /// Returns the added row index.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    pub fn add_row(
        &mut self,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Col, f64)>,
    ) -> Row {
        self.try_add_row(bounds, row_factors)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to add a new constraint to the highs model.
    ///
    /// Returns the added row index, or the error status value if HIGHS returned an error status.
    pub fn try_add_row(
        &mut self,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Col, f64)>,
    ) -> Result<Row, HighsStatus> {
        let (cols, factors): (Vec<_>, Vec<_>) = row_factors.into_iter().unzip();

        unsafe {
            highs_call!(Highs_addRow(
                self.highs.mut_ptr(),
                bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY),
                cols.len().try_into().unwrap(),
                cols.into_iter()
                    .map(|c| c.try_into().unwrap())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                factors.as_ptr()
            ))
        }?;

        Ok(((self.highs.num_rows()? - 1) as c_int).try_into().unwrap())
    }

    /// Adds a new variable to the highs model.
    ///
    /// Returns the added column index.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    pub fn add_col(
        &mut self,
        col_factor: f64,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Row, f64)>,
    ) -> Col {
        self.try_add_column(col_factor, bounds, row_factors)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to add a new variable to the highs model.
    ///
    /// Returns the added column index, or the error status value if HIGHS returned an error status.
    pub fn try_add_column(
        &mut self,
        col_factor: f64,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Row, f64)>,
    ) -> Result<Col, HighsStatus> {
        let (rows, factors): (Vec<_>, Vec<_>) = row_factors.into_iter().unzip();
        unsafe {
            highs_call!(Highs_addCol(
                self.highs.mut_ptr(),
                col_factor,
                bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY),
                rows.len().try_into().unwrap(),
                rows.into_iter()
                    .map(|r| r.try_into().unwrap())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                factors.as_ptr()
            ))
        }?;

        Ok(self.highs.num_cols()? - 1)
    }

    /// Deletes a constraint from the highs model.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    pub fn del_row(&mut self, row: Row) {
        self.try_del_row(row)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to delete a constraint from the highs model.
    ///
    /// Returns an error status value if HIGHS returned an error status.
    pub fn try_del_row(&mut self, row: Row) -> Result<(), HighsStatus> {
        self.try_del_rows(vec![row])
    }

    /// Deletes constraints from the highs model.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    ///
    pub fn del_rows(&mut self, rows: Vec<Row>) {
        self.try_del_rows(rows)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to delete constraints from the highs model.
    ///
    /// Returns an error status value if HIGHS returned an error status.
    pub fn try_del_rows(&mut self, rows: Vec<Row>) -> Result<(), HighsStatus> {
        unsafe {
            highs_call!(Highs_deleteRowsBySet(
                self.highs.mut_ptr(),
                rows.len().try_into().unwrap(),
                rows.into_iter()
                    .map(|r| r.try_into().unwrap())
                    .collect::<Vec<_>>()
                    .as_ptr()
            ))?
        };

        Ok(())
    }

    /// Deletes a variable from the highs model.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    pub fn del_col(&mut self, col: Col) {
        self.try_del_col(col)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to delete a variable from the highs model.
    ///
    /// Returns an error status value if HIGHS returned an error status.
    pub fn try_del_col(&mut self, col: Col) -> Result<(), HighsStatus> {
        self.try_del_cols(vec![col])
    }

    /// Deletes variables from the highs model.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    pub fn del_cols(&mut self, cols: Vec<Col>) {
        self.try_del_cols(cols)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to delete variables from the highs model.
    ///
    /// Returns an error status value if HIGHS returned an error status.
    pub fn try_del_cols(&mut self, cols: Vec<Col>) -> Result<(), HighsStatus> {
        unsafe {
            highs_call!(Highs_deleteColsBySet(
                self.highs.mut_ptr(),
                cols.len().try_into().unwrap(),
                cols.into_iter()
                    .map(|c| c.try_into().unwrap())
                    .collect::<Vec<_>>()
                    .as_ptr()
            ))?
        };

        Ok(())
    }

    /// Tries to change the bounds of constraints from the highs model.
    ///
    /// Returns an error status value if HIGHS returned an error status.
    pub fn try_change_rows_bounds(
        &mut self,
        rows: Vec<Row>,
        bounds: impl RangeBounds<f64>,
    ) -> Result<(), HighsStatus> {
        let size = rows.len();
        unsafe {
            highs_call!(Highs_changeRowsBoundsBySet(
                self.highs.mut_ptr(),
                size.try_into().unwrap(),
                rows.into_iter()
                    .map(|r| r.try_into().unwrap())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                vec![bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY); size].as_ptr(),
                vec![bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY); size].as_ptr()
            ))?
        };

        Ok(())
    }

    /// Tries to change the bounds of constraints from the highs model.
    ///
    /// Returns an error status value if HIGHS returned an error status.
    pub fn try_change_row_bounds(
        &mut self,
        row: Row,
        bounds: impl RangeBounds<f64>,
    ) -> Result<(), HighsStatus> {
        unsafe {
            highs_call!(Highs_changeRowsBoundsBySet(
                self.highs.mut_ptr(),
                1,
                vec![row as c_int].as_ptr(),
                vec![bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY)].as_ptr(),
                vec![bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY)].as_ptr()
            ))?
        };

        Ok(())
    }

    /// Changes the bounds of a constraint from the highs model.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    pub fn change_row_bounds(&mut self, row: Row, bounds: impl RangeBounds<f64>) {
        self.try_change_row_bounds(row, bounds)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Changes the bounds of a variable (column) in the highs model.
    pub fn change_col_bounds(&mut self, col: Col, bounds: impl RangeBounds<f64>) {
        self.try_change_col_bounds(col, bounds)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to change the bounds of a variable (column) in the highs model.
    pub fn try_change_col_bounds(
        &mut self,
        col: Col,
        bounds: impl RangeBounds<f64>,
    ) -> Result<(), HighsStatus> {
        let col_indices = [col as i32];
        unsafe {
            highs_call!(Highs_changeColsBoundsBySet(
                self.highs.mut_ptr(),
                1,
                col_indices.as_ptr(),
                vec![bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY)].as_ptr(),
                vec![bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY)].as_ptr()
            ))?
        };
        Ok(())
    }

    /// Changes the objective coefficient of a variable in the highs model.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    pub fn change_col_cost(&mut self, col: Col, cost: f64) {
        self.try_change_col_cost(col, cost)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to change the objective coefficient of a variable in the highs model.
    /// Returns an error status value if HIGHS returned an error status.
    pub fn try_change_col_cost(&mut self, col: Col, cost: f64) -> Result<(), HighsStatus> {
        let col_indices = [col as i32];
        unsafe {
            highs_call!(Highs_changeColsCostBySet(
                self.highs.mut_ptr(),
                1,
                col_indices.as_ptr(),
                vec![cost].as_ptr()
            ))?
        };
        Ok(())
    }
}

impl From<SolvedModel> for Model {
    fn from(solved: SolvedModel) -> Self {
        Self {
            highs: solved.highs,
        }
    }
}

/// Wrapper around a HiGHS pointer.
#[derive(Debug, Clone)]
pub struct HighsPtr(*mut c_void);

impl Drop for HighsPtr {
    fn drop(&mut self) {
        unsafe { Highs_destroy(self.0) }
    }
}

impl Default for HighsPtr {
    fn default() -> Self {
        Self(unsafe { Highs_create() })
    }
}

/// Trait to give access to methods that are common to both `Model` and `SolvedModel`.
pub trait LikeModel {
    /// Returns the number of columns in the model
    fn num_cols(&self) -> usize;

    /// Returns the name of the variable at the given index.
    fn get_col_name(&self, col: Col) -> Result<String, HighsStatus>;

    /// The status of the solution. Should be Optimal if everything went well
    fn status(&self) -> HighsModelStatus;

    /// Get data associated with multiple adjacent rows from the model
    fn get_rows_by_range(
        &self,
        from_row: Row,
        to_row: Row,
    ) -> Result<
        (
            Vec<f64>,
            Vec<f64>,
            usize,
            Vec<HighsInt>,
            Vec<HighsInt>,
            Vec<f64>,
        ),
        HighsStatus,
    >;

    /// Get rows by range in a more structured format
    /// Returns a vector of row data with bounds and constraint coefficients
    fn get_rows_by_range_structured(
        &self,
        from_row: Row,
        to_row: Row,
    ) -> Result<Vec<RowData>, HighsStatus> {
        let (lower, upper, _num_nz, matrix_start, matrix_index, matrix_value) =
            self.get_rows_by_range(from_row, to_row)?;

        let num_rows = (to_row - from_row + 1) as usize;
        let mut rows = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            let start = matrix_start[i] as usize;
            let end = if i == num_rows - 1 {
                matrix_value.len()
            } else {
                matrix_start[i + 1] as usize
            };

            let mut coefficients = Vec::with_capacity(end - start);
            for j in start..end {
                coefficients.push((matrix_index[j] as Col, matrix_value[j]));
            }

            rows.push(RowData {
                lower_bound: lower[i],
                upper_bound: upper[i],
                coefficients,
            });
        }

        Ok(rows)
    }

    /// Get data for a single row/constraint
    /// Returns the row bounds and constraint coefficients
    fn get_row(&self, row: Row) -> Result<RowData, HighsStatus> {
        let mut row_data = self.get_rows_by_range_structured(row, row)?;
        Ok(row_data.pop().unwrap()) // Safe unwrap since we know we have exactly one row
    }

    /// Writes the model to a file
    fn write(&self, path: &str) -> Result<(), HighsStatus>;

    /// Postsolve a solution
    fn postsolve(&mut self, col_vals: Vec<(Col, f64)>) -> Vec<(Col, f64)>;
}

impl<H: HasHighsPtr> LikeModel for H {
    fn postsolve(&mut self, col_vals: Vec<(Col, f64)>) -> Vec<(Col, f64)> {
        let mut col_vals = col_vals;
        let num_cols = self.num_cols();
        let mut flat_col_vals = vec![0.0; num_cols];
        for (col, val) in col_vals.iter_mut() {
            flat_col_vals[*col] = *val;
        }
        let col_vals_ptr = flat_col_vals.as_mut_ptr();
        println!("input {:?}", flat_col_vals);
        let ret = unsafe {
            Highs_postsolve(
                self.highs_ptr().0,
                col_vals_ptr as *const f64,
                null_mut(),
                null_mut(),
            )
        };
        assert_ne!(ret, STATUS_ERROR, "postsolve failed");
        println!("postsolved: {:?}", flat_col_vals);

        flat_col_vals
            .into_iter()
            .enumerate()
            .filter(|&x| x.1 != 0.)
            .map(|x| (x.0 as Col, x.1))
            .collect::<Vec<_>>()
    }
    fn get_col_name(&self, col: Col) -> Result<String, HighsStatus> {
        let highs_ptr = self.highs_ptr();
        let mut name_buf = vec![0u8; kHighsMaximumStringLength as usize];
        unsafe {
            highs_call!(Highs_getColName(
                highs_ptr.unsafe_mut_ptr(),
                col as i32,
                name_buf.as_mut_ptr() as *mut i8
            ))?;
        }

        let name = name_buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as char)
            .collect::<String>();

        Ok(name)
    }

    fn status(&self) -> HighsModelStatus {
        let model_status = unsafe { Highs_getModelStatus(self.highs_ptr().unsafe_mut_ptr()) };
        HighsModelStatus::try_from(model_status).unwrap()
    }

    /// Get data associated with multiple adjacent rows from the model
    /// Returns (lower_bounds, upper_bounds, num_nz, matrix_start, matrix_index, matrix_value)
    ///
    /// To query the constraint coefficients, this function should be called twice:
    /// 1. First call with matrix_start, matrix_index, and matrix_value as None to get num_nz
    /// 2. Second call with properly allocated vectors to get the actual data
    fn get_rows_by_range(
        &self,
        from_row: Row,
        to_row: Row,
    ) -> Result<
        (
            Vec<f64>,
            Vec<f64>,
            usize,
            Vec<HighsInt>,
            Vec<HighsInt>,
            Vec<f64>,
        ),
        HighsStatus,
    > {
        let num_rows = (to_row - from_row + 1) as usize;

        // First call to get the number of non-zero elements
        let mut lower = vec![0.0; num_rows];
        let mut upper = vec![0.0; num_rows];
        let mut num_nz = 0i32;

        unsafe {
            highs_call!(Highs_getRowsByRange(
                self.highs_ptr().unsafe_mut_ptr(),
                from_row as c_int,
                to_row as c_int,
                &mut (num_rows as c_int),
                lower.as_mut_ptr(),
                upper.as_mut_ptr(),
                &mut num_nz,
                std::ptr::null_mut(), // matrix_start
                std::ptr::null_mut(), // matrix_index
                std::ptr::null_mut()  // matrix_value
            ))?;
        }

        // Second call to get the actual matrix data
        let mut matrix_start = vec![0; num_rows];
        let mut matrix_index = vec![0; num_nz as usize];
        let mut matrix_value = vec![0.0; num_nz as usize];
        let mut num_rows_out = num_rows as c_int;

        unsafe {
            highs_call!(Highs_getRowsByRange(
                self.highs_ptr().unsafe_mut_ptr(),
                from_row as c_int,
                to_row as c_int,
                &mut num_rows_out,
                lower.as_mut_ptr(),
                upper.as_mut_ptr(),
                &mut num_nz,
                matrix_start.as_mut_ptr(),
                matrix_index.as_mut_ptr(),
                matrix_value.as_mut_ptr()
            ))?;
        }

        Ok((
            lower,
            upper,
            num_nz as usize,
            matrix_start,
            matrix_index,
            matrix_value,
        ))
    }

    /// Writes the model to a file
    fn write(&self, path: &str) -> Result<(), HighsStatus> {
        let c_path = CString::new(path).expect("invalid path");
        unsafe {
            highs_call!(Highs_writeModel(
                self.highs_ptr().unsafe_mut_ptr(),
                c_path.as_ptr()
            ))
        }?;

        Ok(())
    }

    fn num_cols(&self) -> usize {
        self.highs_ptr().num_cols().unwrap()
    }
}

impl HighsPtr {
    // To be used instead of unsafe_mut_ptr wherever possible
    #[allow(dead_code)]
    const fn ptr(&self) -> *const c_void {
        self.0
    }

    // Needed until https://github.com/ERGO-Code/HiGHS/issues/479 is fixed
    unsafe fn unsafe_mut_ptr(&self) -> *mut c_void {
        self.0
    }

    fn mut_ptr(&mut self) -> *mut c_void {
        self.0
    }

    /// Prevents writing anything to the standard output when solving the model
    pub fn make_quiet(&mut self) {
        // setting log_file seems to cause a double free in Highs.
        // See https://github.com/rust-or/highs/issues/3
        // self.set_option(&b"log_file"[..], "");
        self.set_option(&b"output_flag"[..], false);
        self.set_option(&b"log_to_console"[..], false);
    }

    /// Set a custom parameter on the model
    pub fn set_option<STR: Into<Vec<u8>>, V: HighsOptionValue>(&mut self, option: STR, value: V) {
        let c_str = CString::new(option).expect("invalid option name");
        let status = unsafe { value.apply_to_highs(self.mut_ptr(), c_str.as_ptr()) };
        try_handle_status(status, "Highs_setOptionValue")
            .expect("An error was encountered in HiGHS.");
    }

    /// Number of variables
    pub fn num_cols(&self) -> Result<usize, TryFromIntError> {
        let n = unsafe { Highs_getNumCols(self.0) };
        n.try_into()
    }

    /// Number of constraints
    pub fn num_rows(&self) -> Result<usize, TryFromIntError> {
        let n = unsafe { Highs_getNumRows(self.0) };
        n.try_into()
    }
}

impl SolvedModel {
    /// Get the solution to the problem
    pub fn get_solution(&self) -> Solution {
        let cols = self.num_cols();
        let rows = self.num_rows();
        let mut colvalue: Vec<f64> = vec![0.; cols];
        let mut coldual: Vec<f64> = vec![0.; cols];
        let mut rowvalue: Vec<f64> = vec![0.; rows];
        let mut rowdual: Vec<f64> = vec![0.; rows];

        // Get the primal and dual solution
        unsafe {
            Highs_getSolution(
                self.highs.unsafe_mut_ptr(),
                colvalue.as_mut_ptr(),
                coldual.as_mut_ptr(),
                rowvalue.as_mut_ptr(),
                rowdual.as_mut_ptr(),
            );
        }

        Solution {
            colvalue,
            coldual,
            rowvalue,
            rowdual,
        }
    }

    /// Number of variables
    pub fn num_cols(&self) -> usize {
        self.highs.num_cols().expect("invalid number of columns")
    }

    /// Number of constraints
    pub fn num_rows(&self) -> usize {
        self.highs.num_rows().expect("invalid number of rows")
    }

    /// Get the basis variables
    pub fn get_basic_vars(&self) -> Vec<BasicVar> {
        let mut basis_ids = vec![0; self.num_rows()];
        unsafe {
            highs_call! {
                Highs_getBasicVariables(self.highs.unsafe_mut_ptr(), basis_ids.as_mut_ptr())
            }
            .map_err(|e| {
                println!("Error while getting basic variables: {:?}", e);
            })
            .unwrap();
        }

        let mut res = Vec::with_capacity(self.num_rows());

        for basis_var in basis_ids.into_iter() {
            if basis_var >= 0 {
                res.push(BasicVar::Col(basis_var as Col));
            } else {
                res.push(BasicVar::Row((-basis_var - 1) as Row));
            }
        }

        res
    }

    /// Get basis status
    pub fn get_basis_status(&self) -> (Vec<BasisStatus>, Vec<BasisStatus>) {
        let mut col_status = vec![kHighsBasisStatusZero; self.num_cols()];
        let mut row_status = vec![kHighsBasisStatusZero; self.num_rows()];
        unsafe {
            highs_call! {
                Highs_getBasis(
                    self.highs.unsafe_mut_ptr(),
                    col_status.as_mut_ptr(),
                    row_status.as_mut_ptr()
                )
            }
            .map_err(|e| {
                println!("Error while getting basis status: {:?}", e);
            })
            .unwrap();
        }

        let col_status = col_status.iter().map(|&s| s.into()).collect();
        let row_status = row_status.iter().map(|&s| s.into()).collect();
        (col_status, row_status)
    }

    /// Get the reduced row
    pub fn get_reduced_row(&self, row: Row) -> (Vec<f64>, Vec<HighsInt>) {
        let mut reduced_row = vec![0.; self.num_cols()];
        let row_non_zeros: *mut HighsInt = &mut 0;
        let mut row_index: Vec<HighsInt> = vec![0; self.num_cols()];
        unsafe {
            highs_call! {
                Highs_getReducedRow(
                    self.highs.unsafe_mut_ptr(),
                    row.try_into().unwrap(),
                    reduced_row.as_mut_ptr(),
                    row_non_zeros,
                    row_index.as_mut_ptr()
                )
            }
            .map_err(|e| {
                println!("Error while getting reduced row: {:?}", e);
            })
            .unwrap();
        }
        let num_nonzeros = unsafe { *row_non_zeros };
        row_index = row_index.into_iter().take(num_nonzeros as usize).collect();

        (reduced_row, row_index)
    }

    /// Get the reduced column
    pub fn get_reduced_column(&self, col: Col) -> (Vec<f64>, Vec<HighsInt>) {
        let mut reduced_col = vec![0.; self.num_rows()];
        let col_non_zeros: *mut HighsInt = &mut 0;
        let mut col_index = vec![0; self.num_rows()];

        unsafe {
            highs_call! {
                Highs_getReducedColumn(
                    self.highs.unsafe_mut_ptr(),
                    col.try_into().unwrap(),
                    reduced_col.as_mut_ptr(),
                    col_non_zeros,
                    col_index.as_mut_ptr()
                )
            }
            .map_err(|e| {
                println!("Error while getting reduced column: {:?}", e);
            })
            .unwrap();
        }

        let num_nonzeros = unsafe { *col_non_zeros };
        col_index = col_index.into_iter().take(num_nonzeros as usize).collect();

        (reduced_col, col_index)
    }

    /// Returns solution to x = B^{-1} * b
    pub fn get_basis_sol(&self, mut b: Vec<f64>) -> (Vec<f64>, Vec<HighsInt>) {
        let mut x = vec![0.; self.num_rows()];
        let solution_num_nz: *mut HighsInt = &mut 0;
        let mut solution_index: Vec<HighsInt> = vec![0; self.num_rows()];
        unsafe {
            highs_call! {
                Highs_getBasisSolve(
                    self.highs.unsafe_mut_ptr(),
                    b.as_mut_ptr(),
                    x.as_mut_ptr(),
                    solution_num_nz,
                    solution_index.as_mut_ptr()
                )
            }
            .map_err(|e| {
                println!("Error while getting basis inverse row: {:?}", e);
            })
            .unwrap();
        }

        let num_nonzeros = unsafe { *solution_num_nz };
        // println!("solution_index: {:?}", solution_index);
        // println!("num_nonzeros: {:?}", num_nonzeros);
        solution_index = solution_index
            .into_iter()
            .take(num_nonzeros as usize)
            .collect();

        (x, solution_index)
    }

    /// Gets a row of the basis inverse matrix B^{-1}
    pub fn get_basis_inverse_row(&self, row: Row) -> (Vec<f64>, Vec<HighsInt>) {
        let mut row_vector = vec![0.; self.num_rows()];
        let row_num_nz: *mut HighsInt = &mut 0;
        let mut row_index: Vec<HighsInt> = vec![0; self.num_rows()];

        unsafe {
            highs_call! {
                Highs_getBasisInverseRow(
                    self.highs.unsafe_mut_ptr(),
                    row.try_into().unwrap(),
                    row_vector.as_mut_ptr(),
                    row_num_nz,
                    row_index.as_mut_ptr()
                )
            }
            .map_err(|e| {
                println!("Error while getting basis inverse row: {:?}", e);
            })
            .unwrap();
        }

        let num_nonzeros = unsafe { *row_num_nz };
        row_index = row_index.into_iter().take(num_nonzeros as usize).collect();

        (row_vector, row_index)
    }

    /// Gets a row of the basis inverse matrix B^{-1}
    pub fn get_basis_inverse_col(&self, col: Col) -> (Vec<f64>, Vec<HighsInt>) {
        let mut col_vector = vec![0.; self.num_rows()];
        let col_num_nz: *mut HighsInt = &mut 0;
        let mut col_index = vec![0; self.num_rows()];

        unsafe {
            highs_call! {
                Highs_getBasisInverseCol(
                    self.highs.unsafe_mut_ptr(),
                    col.try_into().unwrap(),
                    col_vector.as_mut_ptr(),
                    col_num_nz,
                    col_index.as_mut_ptr()
                )
            }
            .map_err(|e| {
                println!("Error while getting basis inverse column: {:?}", e);
            })
            .unwrap();
        }

        let num_nonzeros = unsafe { *col_num_nz };
        col_index = col_index.into_iter().take(num_nonzeros as usize).collect();

        (col_vector, col_index)
    }


    /// Gets the objective value
    pub fn obj_val(&self) -> f64 {
        unsafe { Highs_getObjectiveValue(self.highs.ptr()) }
    }
}

/// Trait for types that can provide access to the underlying HiGHS pointer
pub trait HasHighsPtr {
    /// Get the underlying HiGHS pointer
    fn highs_ptr(&self) -> &HighsPtr;

    /// Get a mutable reference to the underlying HiGHS pointer
    fn highs_mut_ptr(&mut self) -> &mut HighsPtr;
}

impl HasHighsPtr for Model {
    fn highs_ptr(&self) -> &HighsPtr {
        &self.highs
    }

    fn highs_mut_ptr(&mut self) -> &mut HighsPtr {
        &mut self.highs
    }
}

impl HasHighsPtr for SolvedModel {
    fn highs_ptr(&self) -> &HighsPtr {
        &self.highs
    }

    fn highs_mut_ptr(&mut self) -> &mut HighsPtr {
        &mut self.highs
    }
}

/// Concrete values of the solution
#[derive(Clone, Debug)]
pub struct Solution {
    colvalue: Vec<f64>,
    coldual: Vec<f64>,
    rowvalue: Vec<f64>,
    rowdual: Vec<f64>,
}

/// Data for a single row/constraint
#[derive(Clone, Debug)]
pub struct RowData {
    /// Lower bound of the constraint
    pub lower_bound: f64,
    /// Upper bound of the constraint  
    pub upper_bound: f64,
    /// Constraint coefficients as (column_index, coefficient) pairs
    pub coefficients: Vec<(Col, f64)>,
}

impl Solution {
    /// The optimal values for each variables (in the order they were added)
    pub fn columns(&self) -> &[f64] {
        &self.colvalue
    }
    /// The optimal values for each variables in the dual problem (in the order they were added)
    pub fn dual_columns(&self) -> &[f64] {
        &self.coldual
    }
    /// The value of the constraint functions
    pub fn rows(&self) -> &[f64] {
        &self.rowvalue
    }
    /// The value of the constraint functions in the dual problem
    pub fn dual_rows(&self) -> &[f64] {
        &self.rowdual
    }
}

impl Index<Col> for Solution {
    type Output = f64;
    fn index(&self, col: Col) -> &f64 {
        &self.colvalue[col]
    }
}

fn try_handle_status(status: c_int, msg: &str) -> Result<HighsStatus, HighsStatus> {
    let status_enum = HighsStatus::try_from(status)
        .expect("HiGHS returned an unexpected status value. Please report it as a bug to https://github.com/rust-or/highs/issues");
    match status_enum {
        status @ HighsStatus::OK => Ok(status),
        status @ HighsStatus::Warning => {
            log::warn!("HiGHS emitted a warning: {}", msg);
            Ok(status)
        }
        error => Err(error),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
/// The status of a variable/column in the basis
pub enum BasisStatus {
    /// The variable is at its lower bound
    Lower,
    /// The variable is at its upper bound
    Basic,
    /// The variable is at its upper bound
    Upper,
    /// The variable is at zero
    Zero,
}

/// A basic variable is either a column or a row
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasicVar {
    /// Basic variable
    Col(Col),
    /// Basic row
    Row(Row),
}

#[allow(non_upper_case_globals)]
impl From<HighsInt> for BasisStatus {
    fn from(status: HighsInt) -> Self {
        match status {
            kHighsBasisStatusLower => BasisStatus::Lower,
            kHighsBasisStatusBasic => BasisStatus::Basic,
            kHighsBasisStatusUpper => BasisStatus::Upper,
            kHighsBasisStatusZero => BasisStatus::Zero,
            _ => panic!("Invalid basis status"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_coefs(coefs: [f64; 2]) {
        // See: https://github.com/rust-or/highs/issues/5
        let mut problem = RowProblem::new();
        // Minimize x + y subject to x ≥ 0, y ≥ 0.
        let x = problem.add_column(1., -1..);
        let y = problem.add_column(1., 0..);
        problem.add_row(..1, [x, y].iter().copied().zip(coefs)); // 1 ≥ x + c y.
        let solution = problem.optimise(Sense::Minimise).solve().get_solution();
        assert_eq!([-1., 0.], solution.columns());
    }

    #[test]
    fn test_single_zero_coef() {
        test_coefs([1.0, 0.0]);
        test_coefs([0.0, 1.0]);
    }

    #[test]
    fn test_all_zero_coefs() {
        test_coefs([0.0, 0.0])
    }

    #[test]
    fn test_no_zero_coefs() {
        test_coefs([1.0, 1.0])
    }

    #[test]
    fn test_infeasible_empty_row() {
        let mut problem = RowProblem::new();
        let row_factors: &[(Col, f64)] = &[];
        problem.add_row(2..3, row_factors);
        let _ = problem.optimise(Sense::Minimise).try_solve();
    }

    #[test]
    fn test_set_cons_coef() {
        let mut problem = RowProblem::new();
        let col = problem.add_column(1., 0..);
        let row_factors: &[(Col, f64)] = &[];
        let row = problem.add_row(2..3, row_factors);
        problem.set_cons_coef(row, col, 1.0);
        let solved = problem.optimise(Sense::Minimise).solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        let solution = solved.get_solution();
        assert_eq!(solution.columns(), vec![2.0]);
    }

    #[test]
    fn test_add_row_and_col() {
        let mut model = Model::new::<Problem<ColMatrix>>(Problem::default());
        let col = model.add_col(1., 1.0.., vec![]);
        model.add_row(..1.0, vec![(col, 1.0)]);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        let solution = solved.get_solution();
        assert_eq!(solution.columns(), vec![1.0]);

        let mut model = Model::from(solved);
        let new_col = model.add_col(1., ..1.0, vec![]);
        model.add_row(2.0.., vec![(new_col, 1.0)]);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Infeasible);
    }

    #[test]
    fn test_del_row_and_col() {
        let mut model = Model::new::<Problem<ColMatrix>>(Problem::default());
        let col = model.add_col(1., 1.0.., vec![]);
        let row1 = model.add_row(..1.0, vec![(col, 1.0)]);
        let row2 = model.add_row(2.0.., vec![(col, 1.0)]);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Infeasible);

        let mut model = Model::from(solved);
        model.del_rows(vec![row1, row2]);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
    }

    #[test]
    fn test_del_cols() {
        let mut model = Model::new::<Problem<ColMatrix>>(Problem::default());

        let col1 = model.add_col(1.0, 0.0.., vec![]);
        let col2 = model.add_col(1.0, 0.0.., vec![]);

        model.add_row(2.0.., vec![(col1, 1.0), (col2, 1.0)]);

        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);

        let mut model = Model::from(solved);
        model.del_cols(vec![col1, col2]);

        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::ModelEmpty);
    }

    #[test]
    fn test_add_col_after_solve() {
        let mut model = Model::new::<Problem<ColMatrix>>(Problem::default());
        model.set_sense(Sense::Maximise);

        model.add_col(1.0, 0.0..10.0, vec![]);
        let solved = model.solve();

        println!("{:?}", solved.get_solution().columns());

        let mut model = Model::from(solved);
        model.add_col(1.0, 0.0..10.0, vec![]);
        let solved = model.solve();

        println!("{:?}", solved.get_solution().columns());
    }

    #[test]
    fn test_basis_methods() {
        let mut problem = RowProblem::new();
        let x = problem.add_column(1.2, 0..);
        let y = problem.add_column(1.7, 0..);
        problem.add_row(..3000, [x].iter().copied().zip([1.]));
        problem.add_row(..4000, [y].iter().copied().zip([1.]));
        problem.add_row(..5000, [x, y].iter().copied().zip([1., 1.]));
        let model = problem.optimise(Sense::Maximise);
        // model.set_option("presolve", "off");
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        let (col_statuses, row_statuses) = solved.get_basis_status();
        assert_eq!(col_statuses, vec![BasisStatus::Basic, BasisStatus::Basic]);
        assert_eq!(
            row_statuses,
            vec![BasisStatus::Basic, BasisStatus::Upper, BasisStatus::Upper]
        );
        let basic_vars = solved.get_basic_vars();
        println!("\nbasic vars: {:?}", basic_vars);
        assert_eq!(basic_vars.len(), 3);
        for var in basic_vars {
            match var {
                BasicVar::Col(c) => {
                    assert_eq!(col_statuses[c], BasisStatus::Basic);
                }
                BasicVar::Row(r) => {
                    assert_eq!(row_statuses[r], BasisStatus::Basic);
                }
            }
        }

        let basic_rows = solved
            .get_basic_vars()
            .iter()
            .filter_map(|var| {
                if let BasicVar::Row(r) = var {
                    Some(*r)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for row in basic_rows {
            assert_eq!(row_statuses[row], BasisStatus::Basic);
        }

        println!("reduced rows:");
        for i in 0..solved.num_rows() {
            let reduced_row = solved.get_reduced_row(i);
            println!("{:?}", reduced_row);
        }

        println!("reduced cols:");
        for i in 0..solved.num_cols() {
            let reduced_col = solved.get_reduced_column(i);
            println!("{:?}", reduced_col);
        }

        println!("basis inverse rows:");
        for i in 0..solved.num_rows() {
            let basis_inverse_row = solved.get_basis_inverse_row(i);
            println!("{:?}", basis_inverse_row);
        }

        println!("basis sol:");
        let basis_sol = solved.get_basis_sol(vec![3000., 4000., 5000.]);
        println!("{:?}", basis_sol);
        assert_eq!(basis_sol.0.len(), solved.num_rows());
    }

    #[test]
    fn test_get_lp() {
        let mut problem = RowProblem::new();
        let x = problem.add_integer_column(1.2, 0..);
        let y = problem.add_column(1.7, 0..);
        problem.add_row(..3000, [x].iter().copied().zip([1.]));
        problem.add_row(..4000, [y].iter().copied().zip([1.]));
        problem.add_row(..5000, [x, y].iter().copied().zip([1., 1.]));
        let model = problem.optimise(Sense::Maximise);

        let (
            num_col,
            num_row,
            num_nz,
            sense,
            offset,
            col_cost,
            col_lower,
            col_upper,
            row_lower,
            row_upper,
            row_data,
            integrality,
        ) = model.get_row_lp();

        println!("num_col {:?}", num_col);
        println!("num_row {:?}", num_row);
        println!("num_nz {:?}", num_nz);
        println!("sense {:?}", sense);
        println!("offset {:?}", offset);
        println!("col_cost {:?}", col_cost);
        println!("col_lower {:?}", col_lower);
        println!("col_upper {:?}", col_upper);
        println!("row_lower {:?}", row_lower);
        println!("row_upper {:?}", row_upper);
        println!("row_data {:?}", row_data);
        println!("integrality {:?}", integrality);

        assert_eq!(num_col, 2);
        assert_eq!(num_row, 3);
        assert_eq!(num_nz, 4);
        assert_eq!(sense, Sense::Maximise);
        assert_eq!(offset, 0.0);
        assert_eq!(col_cost, vec![1.2, 1.7]);
        assert_eq!(col_lower, vec![0.0, 0.0]);
        assert_eq!(col_upper, vec![f64::INFINITY, f64::INFINITY]);
        assert_eq!(
            row_lower,
            vec![f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY]
        );
        assert_eq!(row_upper, vec![3000.0, 4000.0, 5000.0]);
        assert_eq!(
            row_data,
            vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(0, 1.0), (1, 1.0)]]
        );
        assert_eq!(integrality, vec![VarType::Integer, VarType::Continuous]);
    }

    #[test]
    fn test_presolve() {
        let mut problem = RowProblem::new();
        let x = problem.add_column(1.2, 0..);
        let y = problem.add_column(1.7, 0..);
        problem.add_row(..3000, [x].iter().copied().zip([1.]));
        problem.add_row(..4000, [y].iter().copied().zip([1.]));
        problem.add_row(..5000, [x, y].iter().copied().zip([1., 1.]));
        let mut model = problem.optimise(Sense::Maximise);
        model.presolve();

        let (
            _num_col,
            _num_row,
            _num_nz,
            _sense,
            _offset,
            _col_cost,
            _col_lower,
            _col_upper,
            _row_lower,
            _row_upper,
            _row_data,
            _integrality,
        ) = model.get_row_lp();

        let (
            _num_col,
            num_row,
            _num_nz,
            _sense,
            _offset,
            _col_cost,
            _col_lower,
            _col_upper,
            _row_lower,
            _row_upper,
            _row_data,
            _integrality,
        ) = model.get_presolved_row_lp();
        assert_eq!(num_row, 1);
        println!("{:?}", model.get_presolved_row_lp());

        let mut model = model.solve();
        let col_vals = model.postsolve(vec![(x, 1.0), (y, 3.0)]);
        println!("col_vals {:?}", col_vals);
    }

    #[test]
    fn test_read() {
        let mut model = Model::default();
        model.read("data/simple.mps");

        assert_eq!(model.num_rows(), 2);
        assert_eq!(model.num_cols(), 2);

        let (
            _num_col,
            _num_row,
            _num_nz,
            _sense,
            _offset,
            _col_cost,
            _col_lower,
            _col_upper,
            _row_lower,
            _row_upper,
            row_data,
            _integrality,
        ) = model.get_row_lp();

        for row in row_data.iter() {
            assert_eq!(row.len(), 2);
        }

        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        assert!((solved.obj_val() - -27.0).abs() < 1e-6);
    }

    #[test]
    fn test_read_steininf() {
        let mut model = Model::default();
        // model.read("data/stein9inf.mps");
        model.read("../lio/tests/data/p0201.mps");

        let (
            _num_col,
            _num_row,
            _num_nz,
            _sense,
            _offset,
            _col_cost,
            _col_lower,
            _col_upper,
            _row_lower,
            _row_upper,
            _row_data,
            _integrality,
        ) = model.get_row_lp();

        // println!("num_col {:?}", num_col);
        // let model = model.solve();
        // assert_eq!(model.status(), HighsModelStatus::Infeasible);
    }

    #[test]
    fn test_basis_methods2() {
        let mut problem = RowProblem::new();
        let x = problem.add_column(5.5, 0..);
        let y = problem.add_column(2.1, 0..);
        let z = problem.add_column(2.1, 0..5);
        let za = problem.add_column(2.1, 0..5);
        problem.add_row(..2, vec![(x, -1.), (y, 1.), (z, 1.0), (za, 1.0)]);
        problem.add_row(..17, vec![(x, 8.), (y, 2.)]);
        problem.add_row(..11, [x, y].iter().copied().zip([5., 2.]));
        let model = problem.optimise(Sense::Maximise);
        // model.set_option("presolve", "off");
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        let (col_statuses, row_statuses) = solved.get_basis_status();
        println!("col_statuses: {:?}", col_statuses);
        println!("row_statuses: {:?}", row_statuses);
        let basic_vars = solved.get_basic_vars();
        println!("\nbasic vars: {:?}", basic_vars);

        let mut basic_cols = vec![];
        let mut basic_rows = vec![];
        for var in basic_vars {
            match var {
                BasicVar::Col(c) => {
                    basic_cols.push(c);
                    assert_eq!(col_statuses[c], BasisStatus::Basic);
                }
                BasicVar::Row(r) => {
                    basic_rows.push(r);
                    assert_eq!(row_statuses[r], BasisStatus::Basic);
                }
            }
        }
        println!(
            "\nbasic cols: {:?},\nbasic rows: {:?}",
            basic_cols, basic_rows
        );
        // assert_eq!(basic_cols.len(), 2);
        for col in basic_cols {
            assert_eq!(col_statuses[col], BasisStatus::Basic);
        }
        // assert_eq!(basic_rows.len(), 1);
        for row in basic_rows {
            assert_eq!(row_statuses[row], BasisStatus::Basic);
        }

        println!("reduced rows:");
        for i in 0..solved.num_rows() {
            let reduced_row = solved.get_reduced_row(i);
            println!("{:?}", reduced_row);
        }

        // println!("reduced cols:");
        // for i in 0..solved.num_cols() {
        //     let reduced_col = solved.get_reduced_column(i);
        //     println!("{:?}", reduced_col);
        // }

        println!("basis sol:");
        let basis_sol = solved.get_basis_sol(vec![2., 17., 11.]);
        println!("{:?}", basis_sol);
        assert_eq!(basis_sol.0.len(), solved.num_rows());

        // get the basis inverse row
        for i in 0..solved.num_rows() {
            let basis_inverse_row = solved.get_basis_inverse_row(i);
            println!("basis_inverse_row: {:?}", basis_inverse_row);
        }

        println!("objective value: {:?}", solved.obj_val());
    }

    #[test]
    fn test_get_rows_by_range() {
        let mut problem = RowProblem::new();
        let x = problem.add_column(1.2, 0..);
        let y = problem.add_column(1.7, 0..);
        problem.add_row(..3000, [x].iter().copied().zip([1.]));
        problem.add_row(..4000, [y].iter().copied().zip([1.]));
        problem.add_row(..5000, [x, y].iter().copied().zip([1., 1.]));
        let model = problem.optimise(Sense::Maximise);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);

        // Test get_rows_by_range
        let result = solved.get_rows_by_range(0, 2);
        assert!(result.is_ok());
        let (lower, upper, num_nz, matrix_start, matrix_index, matrix_value) = result.unwrap();

        println!("lower: {:?}", lower);
        println!("upper: {:?}", upper);
        println!("num_nz: {:?}", num_nz);
        println!("matrix_start: {:?}", matrix_start);
        println!("matrix_index: {:?}", matrix_index);
        println!("matrix_value: {:?}", matrix_value);

        assert_eq!(lower.len(), 3);
        assert_eq!(upper, vec![3000.0, 4000.0, 5000.0]);
        assert_eq!(num_nz, 4); // Total non-zero elements across all 3 rows

        // Test structured version
        let structured_result = solved.get_rows_by_range_structured(0, 2);
        assert!(structured_result.is_ok());
        let row_data = structured_result.unwrap();

        assert_eq!(row_data.len(), 3);
        assert_eq!(row_data[0].upper_bound, 3000.0);
        assert_eq!(row_data[1].upper_bound, 4000.0);
        assert_eq!(row_data[2].upper_bound, 5000.0);

        // Check coefficients
        assert_eq!(row_data[0].coefficients, vec![(0, 1.0)]);
        assert_eq!(row_data[1].coefficients, vec![(1, 1.0)]);
        assert_eq!(row_data[2].coefficients, vec![(0, 1.0), (1, 1.0)]);
    }

    #[test]
    fn test_get_row() {
        let mut problem = RowProblem::new();
        let x = problem.add_column(1.2, 0..);
        let y = problem.add_column(1.7, 0..);
        problem.add_row(..3000, [x].iter().copied().zip([1.]));
        problem.add_row(..4000, [y].iter().copied().zip([1.]));
        problem.add_row(..5000, [x, y].iter().copied().zip([1., 1.]));
        let model = problem.optimise(Sense::Maximise);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);

        // Test get_row for each row
        let row0 = solved.get_row(0).unwrap();
        assert_eq!(row0.upper_bound, 3000.0);
        assert_eq!(row0.lower_bound, f64::NEG_INFINITY);
        assert_eq!(row0.coefficients, vec![(0, 1.0)]);

        let row1 = solved.get_row(1).unwrap();
        assert_eq!(row1.upper_bound, 4000.0);
        assert_eq!(row1.lower_bound, f64::NEG_INFINITY);
        assert_eq!(row1.coefficients, vec![(1, 1.0)]);

        let row2 = solved.get_row(2).unwrap();
        assert_eq!(row2.upper_bound, 5000.0);
        assert_eq!(row2.lower_bound, f64::NEG_INFINITY);
        assert_eq!(row2.coefficients, vec![(0, 1.0), (1, 1.0)]);
    }

    #[test]
    fn test_postsolve_with_offset() {
        // Create a simple problem with one variable that has a lower bound of 2
        // This should result in presolve creating an empty problem with an offset
        let mut problem = RowProblem::new();
        let _x = problem.add_column(2.0, 2..); // x >= 2, objective coefficient = 1

        // No constraints - just minimize x subject to x >= 2
        // The optimal solution should be x = 2 with objective value = 2

        let mut model = problem.optimise(Sense::Minimise);

        // Apply presolve
        model.presolve();

        // Check the presolved problem
        let (
            presolved_num_col,
            presolved_num_row,
            presolved_num_nz,
            _presolved_sense,
            presolved_offset,
            _presolved_col_cost,
            _presolved_col_lower,
            _presolved_col_upper,
            _presolved_row_lower,
            _presolved_row_upper,
            _presolved_row_data,
            _presolved_integrality,
        ) = model.get_presolved_row_lp();

        println!("Presolved problem:");
        println!("  Variables: {}", presolved_num_col);
        println!("  Constraints: {}", presolved_num_row);
        println!("  Non-zeros: {}", presolved_num_nz);
        println!("  Offset: {}", presolved_offset);

        // The presolved problem should be empty (or nearly empty) with an offset of 2
        // since the optimal solution is trivially x = 2
        assert_eq!(
            presolved_offset, 4.0,
            "Presolved problem should have offset of 2"
        );

        // Solve the model
        let mut solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);

        // Get the solution
        let solution = solved.get_solution();
        println!("Original solution: {:?}", solution.columns());
        println!("Objective value: {}", solved.obj_val());

        // The solution should be x = 2 (minimum feasible value)
        // The objective value should be 2 * 2 = 4 (coefficient * variable)
        assert!(
            (solution.columns()[0] - 2.0).abs() < 1e-6,
            "Solution should be x = 2, got {}",
            solution.columns()[0]
        );
        assert!(
            (solved.obj_val() - 4.0).abs() < 1e-6,
            "Objective value should be 4, got {}",
            solved.obj_val()
        );

        // Test postsolve with a different value to see if it gets transformed correctly
        let test_value = 5.0;
        let postsolved = solved.postsolve(vec![(0, test_value)]);
        println!(
            "Postsolved result for input {}: {:?}",
            test_value, postsolved
        );

        // The postsolve should return the input value since there's no transformation needed
        assert!(!postsolved.is_empty(), "Postsolve should return a value");
        assert_eq!(postsolved[0].0, 0, "Should be for variable 0");

        // Test with zero value (should be filtered out)
        let postsolved_zero = solved.postsolve(vec![(0, 0.0)]);
        println!("Postsolved result for input 0: {:?}", postsolved_zero);
        assert!(
            postsolved_zero.is_empty(),
            "Zero values should be filtered out"
        );
    }


    #[test]
    fn test_read_and_write() {
        let mut model = Model::default();
        model.read("data/simple.mps");

        assert_eq!(model.num_rows(), 2);
        assert_eq!(model.num_cols(), 2);

        // Write the model to a file
        let write_path = "data/test_write.mps";
        model.write(write_path).unwrap();

        // Read the model back from the file
        let mut read_model = Model::default();
        read_model.read(write_path);

        assert_eq!(read_model.num_rows(), 2);
        assert_eq!(read_model.num_cols(), 2);

        // Clean up the written file
        std::fs::remove_file(write_path).unwrap();
    }
}
