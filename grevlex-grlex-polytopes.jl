# ==============================================================================
#   EdgeExpansionConvADMM -- Code to cmpute the convexified lower bound on
#                            the edge expansion problem.
# ------------------------------------------------------------------------------
#   Copyright (C) 2024 Melanie Siebenhofer <melaniesi@edu.aau.at>
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see https://www.gnu.org/licenses/.
# ==============================================================================

using LinearAlgebra

"""
    grlex(d)

Return the Laplacian matrix of the graph of the grlex polytope in dimension `d`.

For details on the graph of the polytope,
see https://arxiv.org/pdf/1612.06332.pdf, figure 2.
The order of the vertices is θ, u3, ..., ud, w, v12, v13, v23, v14, ..., v1d, ..
.., v{d-1}d, 0

# Copyright
This code is based on a Matlab implementation to get the adjacency matrix
written by Nicolo Gusmeroli (2020).

# Examples
```julia-repl
julia> print(grlex(5))
[5 -1 -1 -1 -1 -1 0 0 0 0 0 0 0 0 0 0; -1 5 -1 -1 -1 0 0 -1 0 0 0 0 0 0 0 0;
-1 -1 6 -1 -1 -1 0 0 0 0 -1 0 0 0 0 0; -1 -1 -1 8 -1 -1 -1 -1 0 0 0 0 0 0 -1 0;
-1 -1 -1 -1 11 -1 -1 -1 -1 -1 -1 0 0 0 0 -1;
-1 0 -1 -1 -1 5 -1 0 0 0 0 0 0 0 0 0; 0 0 0 -1 -1 -1 5 -1 -1 0 0 0 0 0 0 0;
0 -1 0 -1 -1 0 -1 5 0 -1 0 0 0 0 0 0; 0 0 0 0 -1 0 -1 0 5 -1 -1 -1 0 0 0 0;
0 0 0 0 -1 0 0 -1 -1 5 -1 0 -1 0 0 0; 0 0 -1 0 -1 0 0 0 -1 -1 5 0 0 -1 0 0;
0 0 0 0 0 0 0 0 -1 0 0 5 -1 -1 -1 -1; 0 0 0 0 0 0 0 0 0 -1 0 -1 5 -1 -1 -1;
0 0 0 0 0 0 0 0 0 0 -1 -1 -1 5 -1 -1; 0 0 0 -1 0 0 0 0 0 0 0 -1 -1 -1 5 -1;
0 0 0 0 -1 0 0 0 0 0 0 -1 -1 -1 -1 5]
```
"""
function grlex(d)
    # graph from https://arxiv.org/pdf/1612.06332.pdf, figure 2
    v = [d, 1:d-1..., 1]
    p = length(v)
    n = sum(v)
    L = zeros(Int64,n,n)

    # clique between θ, u_k for 3 ⩽ k ⩽ d, w
    #               and v - blocks
    for i=1:p
        a1 = sum(v[1:i])
        a2 = a1 - v[i] + 1
        L[a2:a1,a2:a1] .= -1
    end

    # row-block 1:
    for r=1:d
        q = sum(v[1:r+1])
        L[r,q] = -1    # edge between theta,v_1,2; v_2,3, u_3, ..., v_d-1,d,u_d
                        # and between w and 0
        L[r,d+1:sum(v[1:r-1])] .= -1 # r ⩾ 3, edges of type (7) -> cliques
                                      # between v-cliques and u_k's
    end

    # row-blocks 2:d
    for i=2:p-2
        l = v[i]
        for k = 1:l
            L[sum(v[1:i-1])+k, sum(v[1:i])+k] = -1 # edges from clique v_i to
                                                    # clique v_i+1
        end
    end
    # row-block d+1
    r = sum(v[1:p-2]) + 1
    c = sum(v[1:p])
    L[r:r+v[p-1],c] .= -1 # edges from last v-clique to 0

    # make symmetric matrix! make laplacian!
    #L = UpperTriangular(L)
    L = Symmetric(L,:U)
    for i = 1:n
        L[i,i] -= sum(L[:,i])
    end
    return L
end


"""
    grevlex(d)

Return the Laplacian matrix of the graph of the grevlex polytope
in dimension `d`.

See https://arxiv.org/pdf/1612.06332.pdf, figure 3 for a description
of the graph of the grevlex polytope in dimension `d`
The order of the vertices is u2, u3, ..., u{d+1}, θ, v13, ..., v1{d+1},
v24, ..., v2{d+1}, v35, ..., v{d-2}v{d+1}, v{d-1}{d+1}

# Copyright
This code is based on a Matlab implementation to get the adjacency matrix
written by Nicolo Gusmeroli (2020).

# Examples
```julia-repl
julia> grevlex(4)
11×11 Symmetric{Int64,UpperTriangular{Int64,Array{Int64,2}}}:
  4  -1   0   0   0  -1  -1  -1   0   0   0
 -1   4  -1   0   0   0   0   0  -1  -1   0
  0  -1   4  -1   0  -1   0   0   0   0  -1
  0   0  -1   4  -1   0  -1   0  -1   0   0
  0   0   0  -1   4   0   0  -1   0  -1  -1
 -1   0  -1   0   0   4  -1  -1   0   0   0
 -1   0   0  -1   0  -1   5  -1  -1   0   0
 -1   0   0   0  -1  -1  -1   6   0  -1  -1
  0  -1   0  -1   0   0  -1   0   4  -1   0
  0  -1   0   0  -1   0   0  -1  -1   5  -1
  0   0  -1   0  -1   0   0  -1   0  -1   4
```
"""
function grevlex(d)
    # graph from https://arxiv.org/pdf/1612.06332.pdf, figure 3
    # order:  u2, ... , u{d+1}, 0,
    #         v13, ..., v1{d+1},
    #         v24, ... , v2{d+1},
    #         v35, ... v3{d+1},
    #         v{d-2}{d}, v{d-2}{d+1},
    #         v{d-1}{d+1}
    #   (row-wise)
    v = [d+1,d-1:-1:1 ...]
    n = (d^2 + d + 2) >> 1

    L = zeros(Int64, n,n)
    # row block 1
    # L[2:n+1:n*d] .= -1       # edges u_2-u_3-u_{d+1}-0
    L[n+1:n+1:n*(d+1)] .= -1 # edges u_2-u_3-u_{d+1}-0
    for i=2:d
        r = v[i]
        m = sum(v[1:i-1])
        L[d-r,m+1:m+r] .= -1 # (d-r) = index of u{i+1}, horizontal clique,
                              # edges between u{i+1} and each of the vertices
                              #               v{i}{i+2},...,v{i}{d+1}
        L[m*n+d-r+2:n+1:n*(m+r)] .= -1 # edge between v{i}{j} and u{j+1} for
                                        # all j with  i+2 ⩽ j ⩽ d+1
    end
    # other row blocks
    for j = 2:d
        r = v[j]
        m = sum(v[1:j-1])
        L[m+1:m+r,m+1:m+r] .= -1 # horizontal cliques (v{i}{i+2},...,v{i}{d+1})
        for i=j+1:d
            r2 = v[i]
            m2 = sum(v[1:i-1])
            L[m2*n+m+1+r-r2:n+1:n*(m2+r2)] .= -1 # edge to elem below in row of u{i}
                                                  # for vertical cliques
                                                  # edge betw. v{j-1}. and v{i-1}.
        end
    end
    # make symmetric matrix! make laplacian!
    #L = UpperTriangular(L)
    L = Symmetric(L,:U)
    for i = 1:n
        L[i,i] -= sum(L[:,i])
    end
    return L
end
