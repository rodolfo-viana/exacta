// Configure-time probe run by CMakeLists.txt with the flags of the kernels. It exits with 0 if the
// compiler rounded every product and sum separately, and with 1 if it fused a multiply and an
// add (or kept an intermediate in extended precision), which the kernels' exactness argument
// does not allow. Not part of the library.
//
// With x = 1 + 2^-27, x * x = 1 + 2^-26 + 2^-54 exactly. Rounded to binary64 the product is
// 1 + 2^-26, so the sum with -(1 + 2^-26) is 0; a fused multiply-add keeps the 2^-54 and gives
// 2^-54. The operands are volatile so that no expression is folded at compile time, and each
// form (one expression, or a product stored in a variable and added later) is tried, since
// compilers differ in how far a contraction reaches.

#include <cstdio>

int main() {
  volatile double x = 1.0 + 1.0 / 134217728.0;   // 1 + 2^-27
  volatile double c = -(1.0 + 1.0 / 67108864.0);  // -(1 + 2^-26)
  volatile double d = 1.0 + 1.0 / 67108864.0;    // 1 + 2^-26
  const double xv = x, cv = c, dv = d;
  const double same_expression = xv * xv + cv;
  const double product = xv * xv;
  const double across_statements = product + cv;
  const double subtracted = xv * xv - dv;
  const bool rounded = same_expression == 0.0 && across_statements == 0.0 && subtracted == 0.0;
  if (!rounded) {
    std::printf("fused or extended-precision arithmetic: %.17g %.17g %.17g\n", same_expression,
                across_statements, subtracted);
  }
  return rounded ? 0 : 1;
}
