function plot_err(meshXY,error)   
    % wheather not add the value on boundary points to original solution
    new_solu_err = addBD2func_mesh(meshXY, error);
    
    % plot the solu or err in 3D with dot
%     figure('name','udot')
%     plot3(meshXY(1,:),meshXY(2,:),new_solu_err, 'r.');
%     hold on

    %Ä¿±êÇúÃæµÄ´óÐ¡£¬ÐèÒªÏÈÉú³ÉÒ»¸öÕ¤¸ñ
    % get the coords of X direction and coords of Y direction
    Xcoord = unique(meshXY(1,:));
    Ycoord = unique(meshXY(2,:));
    [Xn,Yn] = meshgrid(Xcoord,Ycoord);

    %ÀûÓÃgriddataÀ´²åÖµ£¬´ÓxyzÉú³ÉÕ¤¸ñÊý¾Ý
    %×îºóÒ»¸öÎª²åÖµ·½·¨£¬°ülinear cubic natural nearestºÍv4µÈ·½·¨
    %v4·½·¨ºÄÊ±³¤µ«½ÏÎª×¼È·
    surf_new_solu_err = griddata(meshXY(1,:),meshXY(2,:),new_solu_err,Xn,Yn,'v4'); 
    surf(Xn,Yn,surf_new_solu_err);
    colorbar;
    caxis([0 5e-5])
%     shading interp

%     figure('name', 'umesh')
%     mesh(Xn,Yn,surf_new_solu_err)
%     colorbar;
%     hold on
    %plot3(meshXY(1,:),meshXY(1,:),u,'r+','MarkerSize',3)
end