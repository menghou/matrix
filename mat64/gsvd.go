// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// Based on the SingularValueDecomposition class from Jama 1.0.3.

package mat64

import (
	"github.com/gonum/blas/blas64"
	"github.com/gonum/floats"
	"github.com/gonum/lapack"
	"github.com/gonum/lapack/lapack64"
	"github.com/gonum/matrix"
)

// GSVD is a type for creating and using the Generalized Singular Value Decomposition
// (GSVD) of a matrix.
type GSVD struct {
	kind matrix.GSVDKind

	m, p, n, k, l int
	c, s          []float64
	a, b, u, v, q blas64.General

	work  []float64
	iwork []int
}

// Factorize computes the generalized singular value decomposition (GSVD) of the input
// the m×n matrix A and the p×n matrix B. The singular values of A and B are computed
// in all cases, while the singular vectors are optionally computed depending on the
// input kind.
//
// The full singular value decomposition (kind == GSVDU|GSVDV|GSVDQ) deconstructs A and B as
//  A = U * Σ₁ * [ 0 R ] * Q^T
//
//  B = V * Σ₂ * [ 0 R ] * Q^T
// where Σ₁ and Σ₂ are m×(k+l) and p×(k+l) diagonal matrices of singular values, and
// U, V and Q are m×m, p×p and n×n orthogonal matrices of singular vectors.
//
// It is frequently not necessary to compute the full GSVD. Computation time and
// storage costs can be reduced using the appropriate kind. Either only the singular
// values can be computed (kind == SVDNone), or in conjunction with specific singular
// vectors (kind bit set according to matrix.GSVDU, matrix.GSVDV and matrix.GSVDQ).
//
// Factorize returns whether the decomposition succeeded. If the decomposition
// failed, routines that require a successful factorization will panic.
func (gsvd *GSVD) Factorize(a, b Matrix, kind matrix.GSVDKind) (ok bool) {
	m, n := a.Dims()
	gsvd.m, gsvd.n = m, n
	p, n := b.Dims()
	gsvd.p = p
	if gsvd.n != n {
		panic(matrix.ErrShape)
	}
	var jobU, jobV, jobQ lapack.GSVDJob
	switch {
	default:
		panic("gsvd: bad input kind")
	case kind == matrix.GSVDNone:
		jobU = lapack.GSVDNone
		jobV = lapack.GSVDNone
		jobQ = lapack.GSVDNone
	case (matrix.GSVDU|matrix.GSVDV|matrix.GSVDQ)&kind != 0:
		if matrix.GSVDU&kind != 0 {
			jobU = lapack.GSVDU
			gsvd.u = blas64.General{
				Rows:   m,
				Cols:   m,
				Stride: m,
				Data:   use(gsvd.u.Data, m*m),
			}
		}
		if matrix.GSVDV&kind != 0 {
			jobV = lapack.GSVDV
			gsvd.v = blas64.General{
				Rows:   p,
				Cols:   p,
				Stride: p,
				Data:   use(gsvd.v.Data, p*p),
			}
		}
		if matrix.GSVDQ&kind != 0 {
			jobQ = lapack.GSVDQ
			gsvd.q = blas64.General{
				Rows:   n,
				Cols:   n,
				Stride: n,
				Data:   use(gsvd.q.Data, n*n),
			}
		}
	}

	// A and B are destroyed on call, so copy the matrices.
	aCopy := DenseCopyOf(a)
	bCopy := DenseCopyOf(b)

	gsvd.c = use(gsvd.c, n)
	gsvd.s = use(gsvd.s, n)

	gsvd.iwork = useInt(gsvd.iwork, n)

	gsvd.work = use(gsvd.work, 1)
	lapack64.Ggsvd3(jobU, jobV, jobQ, aCopy.mat, bCopy.mat, gsvd.c, gsvd.s, gsvd.u, gsvd.v, gsvd.q, gsvd.work, -1, gsvd.iwork)
	gsvd.work = use(gsvd.work, int(gsvd.work[0]))
	gsvd.k, gsvd.l, ok = lapack64.Ggsvd3(jobU, jobV, jobQ, aCopy.mat, bCopy.mat, gsvd.c, gsvd.s, gsvd.u, gsvd.v, gsvd.q, gsvd.work, len(gsvd.work), gsvd.iwork)
	if ok {
		gsvd.a = aCopy.mat
		gsvd.b = bCopy.mat
		gsvd.kind = kind
	}
	return ok
}

// Kind returns the matrix.GSVDKind of the decomposition. If no decomposition has been
// computed, Kind returns 0.
func (gsvd *GSVD) Kind() matrix.GSVDKind {
	return gsvd.kind
}

// Rank returns the k and l terms of the rank of [ A^T B^T ]^T.
func (gsvd *GSVD) Rank() (k, l int) {
	return gsvd.k, gsvd.l
}

// GeneralizedValues returns the generalized singular values of the factorized matrix
// If the input slice is non-nil, the values will be stored in-place into the slice.
// In this case, the slice must have length min(m,n)-k, and GeneralizedValues will
// panic with matrix.ErrSliceLengthMismatch otherwise. If the input slice is nil,
// a new slice of the appropriate length will be allocated and returned.
//
// GeneralizedValues will panic if the receiver does not contain a successful factorization.
func (gsvd *GSVD) GeneralizedValues(v []float64) []float64 {
	if gsvd.kind == 0 {
		panic("gsvd: no decomposition computed")
	}
	if v == nil {
		v = make([]float64, min(gsvd.m, gsvd.n)-gsvd.k)
	}
	if len(v) != min(gsvd.m, gsvd.n)-gsvd.k {
		panic(matrix.ErrSliceLengthMismatch)
	}
	floats.DivTo(v, gsvd.c[gsvd.k:min(gsvd.m, gsvd.n)], gsvd.s[gsvd.k:min(gsvd.m, gsvd.n)])
	return v
}

// Values1 returns the singular values of the factorized matrix.
// If the input slice is non-nil, the values will be stored in-place into the slice.
// In this case, the slice must have length min(m,n)-k, and Values1 will panic with
// matrix.ErrSliceLengthMismatch otherwise. If the input slice is nil,
// a new slice of the appropriate length will be allocated and returned.
//
// Values1 will panic if the receiver does not contain a successful factorization.
func (gsvd *GSVD) Values1(c []float64) []float64 {
	if gsvd.kind == 0 {
		panic("gsvd: no decomposition computed")
	}
	if c == nil {
		c = make([]float64, min(gsvd.m, gsvd.n)-gsvd.k)
	}
	if len(c) != min(gsvd.m, gsvd.n)-gsvd.k {
		panic(matrix.ErrSliceLengthMismatch)
	}
	copy(c, gsvd.c[gsvd.k:min(gsvd.m, gsvd.n)])
	return c
}

// Values2 returns the singular values of the factorized matrix.
// If the input slice is non-nil, the values will be stored in-place into the slice.
// In this case, the slice must have length min(m,n)-k, and Values2 will panic with
// matrix.ErrSliceLengthMismatch otherwise. If the input slice is nil,
// a new slice of the appropriate length will be allocated and returned.
//
// Values2 will panic if the receiver does not contain a successful factorization.
func (gsvd *GSVD) Values2(s []float64) []float64 {
	if gsvd.kind == 0 {
		panic("gsvd: no decomposition computed")
	}
	if s == nil {
		s = make([]float64, min(gsvd.m, gsvd.n)-gsvd.k)
	}
	if len(s) != min(gsvd.m, gsvd.n)-gsvd.k {
		panic(matrix.ErrSliceLengthMismatch)
	}
	copy(s, gsvd.s[gsvd.k:min(gsvd.m, gsvd.n)])
	return s
}

// ZeroRFromGSVD extracts the matrix [ 0 R ] from the singular value decomposition, storing
// the result in-place into the receiver. [ 0 R ] is size (k+l)×n.
func (m *Dense) ZeroRFromGSVD(gsvd *GSVD) {
	if gsvd.kind == 0 {
		panic("gsvd: no decomposition computed")
	}
	m.reuseAsZeroed(gsvd.k+gsvd.l, gsvd.n)
	a := Dense{
		mat:     gsvd.a,
		capRows: gsvd.m,
		capCols: gsvd.n,
	}
	m.Slice(0, min(gsvd.k+gsvd.l, gsvd.m), gsvd.n-gsvd.k-gsvd.l, gsvd.n).(*Dense).
		Copy(a.Slice(0, min(gsvd.k+gsvd.l, gsvd.m), gsvd.n-gsvd.k-gsvd.l, gsvd.n))
	if gsvd.m < gsvd.k+gsvd.l {
		b := Dense{
			mat:     gsvd.b,
			capRows: gsvd.p,
			capCols: gsvd.n,
		}
		m.Slice(gsvd.m, gsvd.k+gsvd.l, gsvd.n+gsvd.m-gsvd.k-gsvd.l, gsvd.n).(*Dense).
			Copy(b.Slice(gsvd.m-gsvd.k, gsvd.l, gsvd.n+gsvd.m-gsvd.k-gsvd.l, gsvd.n))
	}
}

// Sigma1FromGSVD extracts the matrix Σ₁ from the singular value decomposition, storing
// the result in-place into the receiver. Σ₁ is size m×(k+l).
func (m *Dense) Sigma1FromGSVD(gsvd *GSVD) {
	if gsvd.kind == 0 {
		panic("gsvd: no decomposition computed")
	}
	m.reuseAsZeroed(gsvd.m, gsvd.k+gsvd.l)
	for i := 0; i < gsvd.k; i++ {
		m.set(i, i, 1)
	}
	for i := gsvd.k; i < min(gsvd.m, gsvd.k+gsvd.l); i++ {
		m.set(i, i, gsvd.c[i])
	}
}

// Sigma2FromGSVD extracts the matrix Σ₂ from the singular value decomposition, storing
// the result in-place into the receiver. Σ₂ is size p×(k+l).
func (m *Dense) Sigma2FromGSVD(gsvd *GSVD) {
	if gsvd.kind == 0 {
		panic("gsvd: no decomposition computed")
	}
	m.reuseAsZeroed(gsvd.p, gsvd.k+gsvd.l)
	for i := 0; i < min(gsvd.l, gsvd.m-gsvd.k); i++ {
		m.set(i, i+gsvd.k, gsvd.s[gsvd.k+i])
	}
	for i := gsvd.m - gsvd.k; i < gsvd.l; i++ {
		m.set(i, i+gsvd.k, 1)
	}
}

// UFromGSVD extracts the matrix U from the singular value decomposition, storing
// the result in-place into the receiver. U is size m×m.
func (m *Dense) UFromGSVD(gsvd *GSVD) {
	if gsvd.kind&matrix.GSVDU == 0 {
		panic("mat64: improper GSVD kind")
	}
	r := gsvd.u.Rows
	c := gsvd.u.Cols
	m.reuseAs(r, c)

	tmp := &Dense{
		mat:     gsvd.u,
		capRows: r,
		capCols: c,
	}
	m.Copy(tmp)
}

// VFromGSVD extracts the matrix V from the singular value decomposition, storing
// the result in-place into the receiver. V is size p×p.
func (m *Dense) VFromGSVD(gsvd *GSVD) {
	if gsvd.kind&matrix.GSVDV == 0 {
		panic("mat64: improper GSVD kind")
	}
	r := gsvd.v.Rows
	c := gsvd.v.Cols
	m.reuseAs(r, c)

	tmp := &Dense{
		mat:     gsvd.v,
		capRows: r,
		capCols: c,
	}
	m.Copy(tmp)
}

// QFromGSVD extracts the matrix Q from the singular value decomposition, storing
// the result in-place into the receiver. Q is size n×n.
func (m *Dense) QFromGSVD(gsvd *GSVD) {
	if gsvd.kind&matrix.GSVDQ == 0 {
		panic("mat64: improper GSVD kind")
	}
	r := gsvd.q.Rows
	c := gsvd.q.Cols
	m.reuseAs(r, c)

	tmp := &Dense{
		mat:     gsvd.q,
		capRows: r,
		capCols: c,
	}
	m.Copy(tmp)
}
