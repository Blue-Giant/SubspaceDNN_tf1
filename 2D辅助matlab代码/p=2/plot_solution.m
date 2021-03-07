function plot_solution(solu,meshXY)
    dim2variable = size(meshXY, 1);
    num2points = size(meshXY, 2);
    nn = sqrt(num2points);
    if dim2variable==2
        coord_X = unique(meshXY(1,:));
        coord_Y = unique(meshXY(2,:));
    end
    X = [-1.0 coord_X 1.0];
    Y = [-1.0 coord_Y 1.0];
    [meshX, meshY] = meshgrid(X,Y);
    % require u to be a row vector
    if size(solu, 2) == 1
        solu=solu';
    end
    if length(solu)==num2points % if u only defined on interior dofs
        u0 = reshape(solu, nn, nn)';
    u = padarray(u0,[1,1]);
    axis tight;
    
    h1 = surf(meshX, meshY, u);
%     h2 = surf(X, Y, solu);
%     colorbar
%     set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
%     set(gcf, 'Renderer', 'zbuffer');
end