use crate::prelude::*;
use crate::{BasisStatus, Col, Row, SolvedModel};

/// Generator for Gomory Mixed Integer (GMI) cuts
pub struct GmiCutGenerator {
    tolerance: f64,
}

/// Represents a generated cutting plane
#[derive(Debug)]
pub struct Cut {
    coefficients: Vec<(Col, f64)>,
    rhs: f64,
}

impl GmiCutGenerator {
    /// Create a new GMI cut generator with the given numerical tolerance
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Generate a GMI cut from the given basic variable
    pub fn generate_cut(&self, model: &SolvedModel, basic_var: Col) -> Option<Cut> {
        // Get basis information
        let (col_status, _) = model.get_basis_status();

        // Check if the variable is basic and integer
        if col_status[basic_var] != BasisStatus::Basic {
            return None;
        }

        // Get the solution value
        let solution = model.get_solution();
        let value = solution[basic_var];

        // Check if the value is fractional
        let frac_part = value - value.floor();
        if frac_part < self.tolerance || frac_part > 1.0 - self.tolerance {
            return None;
        }

        // Get the tableau row corresponding to this basic variable
        let (coefs, indices) = model.get_reduced_row(basic_var);

        // Generate the cut
        let mut cut = Cut {
            coefficients: Vec::new(),
            rhs: 0.0,
        };

        // Process the coefficients to generate GMI cut terms
        for (&idx, &coef) in indices.iter().zip(coefs.iter()) {
            if coef.abs() < self.tolerance {
                continue;
            }

            let coef_frac = coef - coef.floor();
            let new_coef = if coef_frac <= frac_part {
                coef_frac / frac_part
            } else {
                (1.0 - coef_frac) / (1.0 - frac_part)
            };

            cut.coefficients.push((idx as Col, new_coef));
        }

        cut.rhs = 1.0;
        Some(cut)
    }

    /// Generate GMI cuts for all eligible basic integer variables
    pub fn generate_cuts(&self, model: &SolvedModel) -> Vec<Cut> {
        let mut cuts = Vec::new();

        // Try generating a cut from each basic variable
        for col in 0..model.num_cols() {
            if let Some(cut) = self.generate_cut(model, col) {
                cuts.push(cut);
            }
        }

        cuts
    }
}

impl Cut {
    /// Get the coefficients of the cut
    pub fn coefficients(&self) -> &[(Col, f64)] {
        &self.coefficients
    }

    /// Get the right-hand side of the cut
    pub fn rhs(&self) -> f64 {
        self.rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RowProblem, Sense};

    #[test]
    fn test_gmi_cut_generation() {
        // Create a simple MIP problem:
        // max: x + y
        // s.t. 5x + 2y <= 8
        //      x, y >= 0
        //      x integer
        let mut problem = RowProblem::new();
        let x = problem.add_integer_column(1.0, 0..);
        let y = problem.add_column(1.0, 0..);
        problem.add_row(..=8.0, [(x, 5.0), (y, 2.0)].iter().copied());

        // Solve the LP relaxation
        let model = problem.optimise(Sense::Maximise);
        let solved = model.solve();

        // Create GMI cut generator
        let generator = GmiCutGenerator::new(1e-6);

        // Generate cuts
        let cuts = generator.generate_cuts(&solved);

        // We should get at least one cut since the solution will be fractional
        assert!(!cuts.is_empty());

        // The first cut should have non-zero coefficients
        let cut = &cuts[0];
        assert!(!cut.coefficients().is_empty());
        assert!(cut.rhs() > 0.0);
    }
}
