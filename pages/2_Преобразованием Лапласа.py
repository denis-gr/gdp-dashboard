import streamlit as st
import plotly.graph_objs as go
import numpy as np
from sympy import parse_expr, lambdify, simplify, solve, series, dsolve, Eq, latex, laplace_transform, inverse_laplace_transform

st.title("Решение с помощью преобзования Лапласа")
expression = st.text_input("Введите выражение:", "exp(x) - Integral(exp(x - t) * f(t), (t, 0, x)) - f(x)")

st.expander("Справка").write("""
'sympify', 'SympifyError', 'cacheit', 'Basic', 'Atom',
'preorder_traversal', 'S', 'Expr', 'AtomicExpr', 'UnevaluatedExpr',
'Symbol', 'Wild', 'Dummy', 'symbols', 'var', 'Number', 'Float',
'Rational', 'Integer', 'NumberSymbol', 'RealNumber', 'igcd', 'ilcm',
'seterr', 'E', 'I', 'nan', 'oo', 'pi', 'zoo', 'AlgebraicNumber', 'comp',
'mod_inverse', 'Pow', 'integer_nthroot', 'integer_log', 'trailing', 'Mul', 'prod',
'Add', 'Mod', 'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'Equality',
'GreaterThan', 'LessThan', 'Unequality', 'StrictGreaterThan',
'StrictLessThan', 'vectorize', 'Lambda', 'WildFunction', 'Derivative',
'diff', 'FunctionClass', 'Function', 'Subs', 'expand', 'PoleError',
'count_ops', 'expand_mul', 'expand_log', 'expand_func', 'expand_trig',
'expand_complex', 'expand_multinomial', 'nfloat', 'expand_power_base',
'expand_power_exp', 'arity', 'PrecisionExhausted', 'N', 'evalf', 'Tuple',
'Dict', 'gcd_terms', 'factor_terms', 'factor_nc', 'evaluate', 'Catalan',
'EulerGamma', 'GoldenRatio', 'TribonacciConstant', 'bottom_up', 'use',
'postorder_traversal', 'default_sort_key', 'ordered', 'num_digits',

# sympy.logic
'to_cnf', 'to_dnf', 'to_nnf', 'And', 'Or', 'Not', 'Xor', 'Nand', 'Nor',
'Implies', 'Equivalent', 'ITE', 'POSform', 'SOPform', 'simplify_logic',
'bool_map', 'true', 'false', 'satisfiable',

# sympy.assumptions
'AppliedPredicate', 'Predicate', 'AssumptionsContext', 'assuming', 'Q',
'ask', 'register_handler', 'remove_handler', 'refine',

# sympy.polys
'Poly', 'PurePoly', 'poly_from_expr', 'parallel_poly_from_expr', 'degree',
'total_degree', 'degree_list', 'LC', 'LM', 'LT', 'pdiv', 'prem', 'pquo',
'pexquo', 'div', 'rem', 'quo', 'exquo', 'half_gcdex', 'gcdex', 'invert',
'subresultants', 'resultant', 'discriminant', 'cofactors', 'gcd_list',
'gcd', 'lcm_list', 'lcm', 'terms_gcd', 'trunc', 'monic', 'content',
'primitive', 'compose', 'decompose', 'sturm', 'gff_list', 'gff',
'sqf_norm', 'sqf_part', 'sqf_list', 'sqf', 'factor_list', 'factor',
'intervals', 'refine_root', 'count_roots', 'all_roots', 'real_roots',
'nroots', 'ground_roots', 'nth_power_roots_poly', 'cancel', 'reduced',
'groebner', 'is_zero_dimensional', 'GroebnerBasis', 'poly', 'symmetrize',
'horner', 'interpolate', 'rational_interpolate', 'viete', 'together',
'BasePolynomialError', 'ExactQuotientFailed', 'PolynomialDivisionFailed',
'OperationNotSupported', 'HeuristicGCDFailed', 'HomomorphismFailed',
'IsomorphismFailed', 'ExtraneousFactors', 'EvaluationFailed',
'RefinementFailed', 'CoercionFailed', 'NotInvertible', 'NotReversible',
'NotAlgebraic', 'DomainError', 'PolynomialError', 'UnificationFailed',
'GeneratorsError', 'GeneratorsNeeded', 'ComputationFailed',
'UnivariatePolynomialError', 'MultivariatePolynomialError',
'PolificationFailed', 'OptionError', 'FlagError', 'minpoly',
'minimal_polynomial', 'primitive_element', 'field_isomorphism',
'to_number_field', 'isolate', 'round_two', 'prime_decomp',
'prime_valuation', 'galois_group', 'itermonomials', 'Monomial', 'lex', 'grlex',
'grevlex', 'ilex', 'igrlex', 'igrevlex', 'CRootOf', 'rootof', 'RootOf',
'ComplexRootOf', 'RootSum', 'roots', 'Domain', 'FiniteField',
'IntegerRing', 'RationalField', 'RealField', 'ComplexField',
'PythonFiniteField', 'GMPYFiniteField', 'PythonIntegerRing',
'GMPYIntegerRing', 'PythonRational', 'GMPYRationalField',
'AlgebraicField', 'PolynomialRing', 'FractionField', 'ExpressionDomain',
'FF_python', 'FF_gmpy', 'ZZ_python', 'ZZ_gmpy', 'QQ_python', 'QQ_gmpy',
'GF', 'FF', 'ZZ', 'QQ', 'ZZ_I', 'QQ_I', 'RR', 'CC', 'EX', 'EXRAW',
'construct_domain', 'swinnerton_dyer_poly', 'cyclotomic_poly',
'symmetric_poly', 'random_poly', 'interpolating_poly', 'jacobi_poly',
'chebyshevt_poly', 'chebyshevu_poly', 'hermite_poly', 'hermite_prob_poly',
'legendre_poly', 'laguerre_poly', 'apart', 'apart_list', 'assemble_partfrac_list',
'Options', 'ring', 'xring', 'vring', 'sring', 'field', 'xfield', 'vfield',
'sfield',

# sympy.series
'Order', 'O', 'limit', 'Limit', 'gruntz', 'series', 'approximants',
'residue', 'EmptySequence', 'SeqPer', 'SeqFormula', 'sequence', 'SeqAdd',
'SeqMul', 'fourier_series', 'fps', 'difference_delta', 'limit_seq',

# sympy.functions
'factorial', 'factorial2', 'rf', 'ff', 'binomial', 'RisingFactorial',
'FallingFactorial', 'subfactorial', 'carmichael', 'fibonacci', 'lucas',
'motzkin', 'tribonacci', 'harmonic', 'bernoulli', 'bell', 'euler', 'catalan',
'genocchi', 'andre', 'partition',  'divisor_sigma', 'legendre_symbol', 'jacobi_symbol',
'kronecker_symbol', 'mobius', 'primenu', 'primeomega', 'totient', 'primepi',
'reduced_totient', 'sqrt', 'root', 'Min', 'Max', 'Id', 'real_root',
'Rem', 'cbrt', 're', 'im', 'sign', 'Abs', 'conjugate', 'arg', 'polar_lift',
'periodic_argument', 'unbranched_argument', 'principal_branch',
'transpose', 'adjoint', 'polarify', 'unpolarify', 'sin', 'cos', 'tan',
'sec', 'csc', 'cot', 'sinc', 'asin', 'acos', 'atan', 'asec', 'acsc',
'acot', 'atan2', 'exp_polar', 'exp', 'ln', 'log', 'LambertW', 'sinh',
'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh',
'acoth', 'asech', 'acsch', 'floor', 'ceiling', 'frac', 'Piecewise',
'piecewise_fold', 'piecewise_exclusive', 'erf', 'erfc', 'erfi', 'erf2',
'erfinv', 'erfcinv', 'erf2inv', 'Ei', 'expint', 'E1', 'li', 'Li', 'Si',
'Ci', 'Shi', 'Chi', 'fresnels', 'fresnelc', 'gamma', 'lowergamma',
'uppergamma', 'polygamma', 'loggamma', 'digamma', 'trigamma', 'multigamma',
'dirichlet_eta', 'zeta', 'lerchphi', 'polylog', 'stieltjes', 'Eijk', 'LeviCivita',
'KroneckerDelta', 'SingularityFunction', 'DiracDelta', 'Heaviside',
'bspline_basis', 'bspline_basis_set', 'interpolating_spline', 'besselj',
'bessely', 'besseli', 'besselk', 'hankel1', 'hankel2', 'jn', 'yn',
'jn_zeros', 'hn1', 'hn2', 'airyai', 'airybi', 'airyaiprime',
'airybiprime', 'marcumq', 'hyper', 'meijerg', 'appellf1', 'legendre',
'assoc_legendre', 'hermite', 'hermite_prob', 'chebyshevt', 'chebyshevu',
'chebyshevu_root', 'chebyshevt_root', 'laguerre', 'assoc_laguerre',
'gegenbauer', 'jacobi', 'jacobi_normalized', 'Ynm', 'Ynm_c', 'Znm',
'elliptic_k', 'elliptic_f', 'elliptic_e', 'elliptic_pi', 'beta',
'mathieus', 'mathieuc', 'mathieusprime', 'mathieucprime', 'riemann_xi','betainc',
'betainc_regularized',

# sympy.ntheory
'nextprime', 'prevprime', 'prime', 'primerange', 'randprime',
'Sieve', 'sieve', 'primorial', 'cycle_length', 'composite', 'compositepi',
'isprime', 'divisors', 'proper_divisors', 'factorint', 'multiplicity',
'perfect_power', 'pollard_pm1', 'pollard_rho', 'primefactors',
'divisor_count', 'proper_divisor_count',
'factorrat',
'mersenne_prime_exponent', 'is_perfect', 'is_mersenne_prime',
'is_abundant', 'is_deficient', 'is_amicable', 'is_carmichael', 'abundance',
'npartitions',
'is_primitive_root', 'is_quad_residue',
'n_order', 'sqrt_mod', 'quadratic_residues',
'primitive_root', 'nthroot_mod', 'is_nthpow_residue', 'sqrt_mod_iter',
'discrete_log', 'quadratic_congruence', 'binomial_coefficients',
'binomial_coefficients_list', 'multinomial_coefficients',
'continued_fraction_periodic', 'continued_fraction_iterator',
'continued_fraction_reduce', 'continued_fraction_convergents',
'continued_fraction', 'egyptian_fraction',

# sympy.concrete
'product', 'Product', 'summation', 'Sum',

# sympy.discrete
'fft', 'ifft', 'ntt', 'intt', 'fwht', 'ifwht', 'mobius_transform',
'inverse_mobius_transform', 'convolution', 'covering_product',
'intersecting_product',

# sympy.simplify
'simplify', 'hypersimp', 'hypersimilar', 'logcombine', 'separatevars',
'posify', 'besselsimp', 'kroneckersimp', 'signsimp',
'nsimplify', 'FU', 'fu', 'sqrtdenest', 'cse', 'epath', 'EPath',
'hyperexpand', 'collect', 'rcollect', 'radsimp', 'collect_const',
'fraction', 'numer', 'denom', 'trigsimp', 'exptrigsimp', 'powsimp',
'powdenest', 'combsimp', 'gammasimp', 'ratsimp', 'ratsimpmodprime',

# sympy.sets
'Set', 'Interval', 'Union', 'EmptySet', 'FiniteSet', 'ProductSet',
'Intersection', 'imageset', 'DisjointUnion', 'Complement', 'SymmetricDifference',
'ImageSet', 'Range', 'ComplexRegion', 'Reals', 'Contains', 'ConditionSet',
'Ordinal', 'OmegaPower', 'ord0', 'PowerSet', 'Naturals',
'Naturals0', 'UniversalSet', 'Integers', 'Rationals', 'Complexes',

# sympy.solvers
'solve', 'solve_linear_system', 'solve_linear_system_LU',
'solve_undetermined_coeffs', 'nsolve', 'solve_linear', 'checksol',
'det_quick', 'inv_quick', 'check_assumptions', 'failing_assumptions',
'diophantine', 'rsolve', 'rsolve_poly', 'rsolve_ratio', 'rsolve_hyper',
'checkodesol', 'classify_ode', 'dsolve', 'homogeneous_order',
'solve_poly_system', 'solve_triangulated', 'pde_separate',
'pde_separate_add', 'pde_separate_mul', 'pdsolve', 'classify_pde',
'checkpdesol', 'ode_order', 'reduce_inequalities',
'reduce_abs_inequality', 'reduce_abs_inequalities',
'solve_poly_inequality', 'solve_rational_inequalities',
'solve_univariate_inequality', 'decompogen', 'solveset', 'linsolve',
'linear_eq_to_matrix', 'nonlinsolve', 'substitution',

# sympy.matrices
'ShapeError', 'NonSquareMatrixError', 'GramSchmidt', 'casoratian', 'diag',
'eye', 'hessian', 'jordan_cell', 'list2numpy', 'matrix2numpy',
'matrix_multiply_elementwise', 'ones', 'randMatrix', 'rot_axis1',
'rot_axis2', 'rot_axis3', 'symarray', 'wronskian', 'zeros',
'MutableDenseMatrix', 'DeferredVector', 'MatrixBase', 'Matrix',
'MutableMatrix', 'MutableSparseMatrix', 'banded', 'ImmutableDenseMatrix',
'ImmutableSparseMatrix', 'ImmutableMatrix', 'SparseMatrix', 'MatrixSlice',
'BlockDiagMatrix', 'BlockMatrix', 'FunctionMatrix', 'Identity', 'Inverse',
'MatAdd', 'MatMul', 'MatPow', 'MatrixExpr', 'MatrixSymbol', 'Trace',
'Transpose', 'ZeroMatrix', 'OneMatrix', 'blockcut', 'block_collapse',
'matrix_symbols', 'Adjoint', 'hadamard_product', 'HadamardProduct',
'HadamardPower', 'Determinant', 'det', 'diagonalize_vector', 'DiagMatrix',
'DiagonalMatrix', 'DiagonalOf', 'trace', 'DotProduct',
'kronecker_product', 'KroneckerProduct', 'PermutationMatrix',
'MatrixPermute', 'Permanent', 'per', 'rot_ccw_axis1', 'rot_ccw_axis2',
'rot_ccw_axis3', 'rot_givens',

# sympy.geometry
'Point', 'Point2D', 'Point3D', 'Line', 'Ray', 'Segment', 'Line2D',
'Segment2D', 'Ray2D', 'Line3D', 'Segment3D', 'Ray3D', 'Plane', 'Ellipse',
'Circle', 'Polygon', 'RegularPolygon', 'Triangle', 'rad', 'deg',
'are_similar', 'centroid', 'convex_hull', 'idiff', 'intersection',
'closest_points', 'farthest_points', 'GeometryError', 'Curve', 'Parabola',

# sympy.utilities
'flatten', 'group', 'take', 'subsets', 'variations', 'numbered_symbols',
'cartes', 'capture', 'dict_merge', 'prefixes', 'postfixes', 'sift',
'topological_sort', 'unflatten', 'has_dups', 'has_variety', 'reshape',
'rotations', 'filldedent', 'lambdify', 'threaded', 'xthreaded',
'public', 'memoize_property', 'timed',

# sympy.integrals
'integrate', 'Integral', 'line_integrate', 'mellin_transform',
'inverse_mellin_transform', 'MellinTransform', 'InverseMellinTransform',
'laplace_transform', 'inverse_laplace_transform', 'LaplaceTransform',
'laplace_correspondence', 'laplace_initial_conds',
'InverseLaplaceTransform', 'fourier_transform',
'inverse_fourier_transform', 'FourierTransform',
'InverseFourierTransform', 'sine_transform', 'inverse_sine_transform',
'SineTransform', 'InverseSineTransform', 'cosine_transform',
'inverse_cosine_transform', 'CosineTransform', 'InverseCosineTransform',
'hankel_transform', 'inverse_hankel_transform', 'HankelTransform',
'InverseHankelTransform', 'singularityintegrate',

# sympy.tensor
'IndexedBase', 'Idx', 'Indexed', 'get_contraction_structure',
'get_indices', 'shape', 'MutableDenseNDimArray', 'ImmutableDenseNDimArray',
'MutableSparseNDimArray', 'ImmutableSparseNDimArray', 'NDimArray',
'tensorproduct', 'tensorcontraction', 'tensordiagonal', 'derive_by_array',
'permutedims', 'Array', 'DenseNDimArray', 'SparseNDimArray',

# sympy.parsing
'parse_expr',

# sympy.calculus
'euler_equations', 'singularities', 'is_increasing',
'is_strictly_increasing', 'is_decreasing', 'is_strictly_decreasing',
'is_monotonic', 'finite_diff_weights', 'apply_finite_diff',
'differentiate_finite', 'periodicity', 'not_empty_in',
'AccumBounds', 'is_convex', 'stationary_points', 'minimum', 'maximum',

# sympy.algebras
'Quaternion',

# sympy.printing
'pager_print', 'pretty', 'pretty_print', 'pprint', 'pprint_use_unicode',
'pprint_try_use_unicode', 'latex', 'print_latex', 'multiline_latex',
'mathml', 'print_mathml', 'python', 'print_python', 'pycode', 'ccode',
'print_ccode', 'smtlib_code', 'glsl_code', 'print_glsl', 'cxxcode', 'fcode',
'print_fcode', 'rcode', 'print_rcode', 'jscode', 'print_jscode',
'julia_code', 'mathematica_code', 'octave_code', 'rust_code', 'print_gtk',
'preview', 'srepr', 'print_tree', 'StrPrinter', 'sstr', 'sstrrepr',
'TableForm', 'dotprint', 'maple_code', 'print_maple_code',

# sympy.plotting
'plot', 'textplot', 'plot_backends', 'plot_implicit', 'plot_parametric',

# sympy.interactive
'init_session', 'init_printing', 'interactive_traversal',

# sympy.testing
'test', 'doctest',
""")

try:
    expr = parse_expr(expression)
except Exception as e:
    st.error(f"Ошибка в выражении: {e}")
    expr = parse_expr("x")
st.latex(latex(expr))
st.code(expr)

st.markdown("##### Преобразование Лапласа")
L_exp = laplace_transform(expr, parse_expr("x"), parse_expr("s"))[0]
st.latex(latex(L_exp))
st.code(L_exp)

st.markdown("##### Ручное преобзование")
L_exp = st.text_input("Введите выражение:", L_exp)
L_exp = solve(L_exp, parse_expr("LaplaceTransform(f(x), x, s)"))[0]
st.latex(latex(L_exp))
st.code(L_exp)
L_exp = L_exp.apart()
st.latex(latex(L_exp))
st.code(L_exp)

st.markdown("##### Ответ")
ans = inverse_laplace_transform(L_exp, parse_expr("s"), parse_expr("x")).subs(parse_expr("Heaviside(x)"), 1)
st.latex(latex(ans))
st.code(ans)
