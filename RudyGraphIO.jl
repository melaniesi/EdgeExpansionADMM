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

module RudyGraphIO

using LinearAlgebra

export laplacian_from_RudyFile, laplacian_to_RudyFile, laplacian_to_graphFile



#--------------------#
#     I N P U T      #
#--------------------#


"""
    laplacian_from_RudyFile(filepath; <keyword arguments>)

Return the Laplacian matrix of the graph
stored in rudy file format in `filepath`.

# Arguments
 - `weights=false`: if `false`, we consider the rudy file
                    format without weights (i.e., in each row
                    there is only a pair of adjacent vertices
                    without a third value, the weight)
"""
function laplacian_from_RudyFile(filepath; weights=false)
    A = weights ? adjacency_from_RudyFile(filepath) : adjacency_from_RudyFileWithoutWeights(filepath)
    return diagm(vec(sum(A, dims=2))) - A   # L = Diag(Ae) - A
end

"""
    adjacency_from_RudyFile(filepath)

Return the adjacency matrix of the graph
stored in rudy file format in `filepath`.

In the first row of the file is the number of vertices and the
number of edges separated by a blank.
Then for each edge {i,j}, there is a line "i j weight".
"""
function adjacency_from_RudyFile(filepath)
    io = open(filepath)
    n, m = parse.(Int64, split(readline(io)," "))
    A = zeros(Int64, n,n) #spzeros(Int64,n,n)
    while !eof(io)
        line = readline(io)
        line = split(line, " ", keepempty=false)
        if !isempty(line) 
            i, j, val = parse.(Int64, line)
            A[i,j] = val
            A[j,i] = val
        end
    end
    close(io)
    return A
end


"""
    adjacency_from_RudyFileWithoutWeights(filepath)

Return the adjacency matrix of the graph
stored in rudy file format without weights.

Without weights means that in each row (except the first) of file 
there is only the pair of adjacent vertices without a weight.
If there is a third entry (weight), it is ignored.
In the first row of the file is the number of vertices and the
number of edges separated by a blank.
"""
function adjacency_from_RudyFileWithoutWeights(filepath)
    io = open(filepath)
    n, m = parse.(Int64, split(readline(io)," "))
    A = zeros(Int64, n,n) #spzeros(Int64,n,n)
    while !eof(io)
        line = readline(io)
        if !isempty(line) 
            i, j = parse.(Int64, split(line, " "))
            if i!=j
                A[i,j] = 1
                A[j,i] = 1
            end
        end
    end
    close(io)
    return A
end

#--------------------#
#    O U T P U T     #
#--------------------#

"""
    laplacian_to_RudyFile(L, filepath)
    
Write to `filepath` the rudy file of
the graph given by its Laplacian matrix `L`.

In the first row of the file is the number of vertices and the
number of edges separated by a blank.
Then for each edge {i,j}, there is a line "i j weight".
"""
function laplacian_to_RudyFile(L, filepath)
    io = open(filepath, "w")
    n = size(L,1)
    A = diagm(diag(L)) - L
    m = Int(count(!iszero, A)/2) # m = #non-zero entries/2
    println(io, "$n $m")
    for i = 1:n
        for j = i+1:n
            if A[i,j] != 0
                println(io, "$i $j $(A[i,j])")
            end
        end
    end
    #write(io, s)
    close(io)
end

"""
    laplacian_to_graphFile(L, filepath)

Write to `filepath` the graph file of
the graph given by the Laplacian matrix `L`.

This file format has in the first line of the document
the number of vertices and edges of the graph followed
by a list of edges in the form (i,j).
"""
function laplacian_to_graphFile(L, filepath)
    io = open(filepath, "w")
    n = size(L,1)
    A = diagm(diag(L)) - L
    m = Int(count(!iszero, A) / 2)
    println(io, "$n $m")
    for i = 1:n
        for j = i+1:n
            if A[i,j] != 0
                println(io, "($i,$j)")
            end
        end
    end
end

end
