function surfAndPlot_U(meshXY, u)
    % get the coords of X direction and coords of Y direction
    Xcoord = unique(meshXY(1,:));
    Ycoord = unique(meshXY(2,:));
    [Xn,Yn] = meshgrid(Xcoord,Ycoord);

    surf_u = griddata(meshXY(1,:),meshXY(2,:),u,Xn,Yn,'v4'); 
    surf(Xn,Yn,surf_u);
    colorbar;
end