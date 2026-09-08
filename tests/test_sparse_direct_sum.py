from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from matrix_smith import (
    SPARSE_B_DIRECT_SUM_CERTIFICATE_SCHEMA,
    SparseBDirectSumCertificateError,
    SparseBMatrix,
    SparseBRow,
    certify_sparse_b_direct_sum,
    verify_sparse_b_direct_sum,
)


def _decomposable_matrix() -> SparseBMatrix:
    return SparseBMatrix.from_dense(
        np.asarray(
            [
                [1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 0.0, -1.0, 0.0],
            ]
        ),
        backend="test-exact-direct-sum",
    )


def test_certifies_support_components_and_isolated_columns() -> None:
    matrix = _decomposable_matrix()
    certificate = certify_sparse_b_direct_sum(matrix)

    assert certificate.schema == SPARSE_B_DIRECT_SUM_CERTIFICATE_SCHEMA
    assert certificate.is_nontrivial
    assert certificate.component_count == 2
    assert certificate.largest_row_block == 2
    assert certificate.largest_column_block == 2
    assert certificate.components[0].row_indices == (0, 1)
    assert certificate.components[0].column_indices == (0, 1)
    assert certificate.components[1].row_indices == (2,)
    assert certificate.components[1].column_indices == (3, 5)
    assert certificate.isolated_column_indices == (2, 4, 6)
    verify_sparse_b_direct_sum(matrix, certificate)


def test_certificate_fails_closed_for_matrix_or_identity_change() -> None:
    matrix = _decomposable_matrix()
    certificate = certify_sparse_b_direct_sum(matrix)
    changed = SparseBMatrix.from_dense(
        matrix.to_dense()
        + np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )

    with pytest.raises(SparseBDirectSumCertificateError, match="does not match"):
        verify_sparse_b_direct_sum(changed, certificate)
    with pytest.raises(SparseBDirectSumCertificateError, match="does not match"):
        verify_sparse_b_direct_sum(
            matrix,
            replace(certificate, matrix_identity_sha256="0" * 64),
        )


def test_automatic_partition_storage_is_linear() -> None:
    coordinate_count = 2000
    matrix = SparseBMatrix(
        rows=tuple(
            SparseBRow(
                size=2 * coordinate_count,
                entries=((2 * index, 1.0), (2 * index + 1, -1.0)),
            )
            for index in range(coordinate_count)
        ),
        column_count=2 * coordinate_count,
    )
    certificate = certify_sparse_b_direct_sum(matrix)

    assert matrix.nnz == 2 * coordinate_count
    assert certificate.component_count == coordinate_count
    assert certificate.largest_row_block == 1
    assert certificate.largest_column_block == 2
