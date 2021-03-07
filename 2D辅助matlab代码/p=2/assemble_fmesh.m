function geom = assemble_fmesh(geom)
%ASSEMBLE_FMESH generate fine mesh for [-1, 1] x [-1, 1]
% this requires the side length geom.L = (geom.N1+1)*geom.h

if nargin == 0
    test_assemble_fmesh;
    return
end
% [x, y] = meshgrid(0:geom.h:1, 0:geom.h:1);
[x, y] = meshgrid(-1:geom.h:1, -1:geom.h:1);
geom.X = x;
geom.Y = y;

x = x'; x = x(:)'; 
y = y'; y = y(:)';

p = [x; y];

% create the element matrix t for square elements
N = geom.N1 + 1;
t = zeros(4, N^2);
nb = speye(N^2, N^2);
for i = 1:N
    for j = 1:N
        ind = i + N*(j-1);
        t(:, ind) = [
            (j-1)*(N+1)+i;
            (j-1)*(N+1)+i+1;
            j*(N+1)+i;
            j*(N+1)+i+1;
            ];
        % pt is the center of elements t
        pt(:, ind) = sum(p(:, t(:, ind)), 2)/4;
    end
end

% identify interior dofs and boundary dofs
ib = abs(x)==1 | abs(y)==1;
in = not(ib);
pin = p(:, in);

% [ind, ~, ~, adjmatrix, ~, ~] = quadtree(pin(1, :), pin(2, :), [], 1);
% ind is the index of regions
% bx, by denotes the order in x and y direction respectively
% nb is the adjacency matrix of regions
ind = quadtreeind(geom.q);

% now reorder the dofs by the regions, this create a natural order
% multiresolution operations by using the base 4 representation of indices
pin(:, ind) = pin;
nb = sparse(nb);

geom.p = p;
geom.t = t;
geom.pt = pt;
geom.ib = ib;
geom.in = in;
geom.ind = ind;
geom.pin = pin;
% geom.adjmatrix = adjmatrix;

end

