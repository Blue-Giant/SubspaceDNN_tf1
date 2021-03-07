function plot_solu(meshXY,solu)   
    % wheather not add the value on boundary points to original solution
    new_solu = addBD2func_mesh(meshXY, solu);
    
    % plot the solution in 3D with dot
%     figure('name','udot')
%     plot3(meshXY(1,:),meshXY(2,:),new_solu, 'r.');
%     hold on

    %Ä¿±êÇúÃæµÄ´óÐ¡£¬ÐèÒªÏÈÉú³ÉÒ»¸öÕ¤¸ñ
    % get the coords of X direction and coords of Y direction
    Xcoord = unique(meshXY(1,:));
    Ycoord = unique(meshXY(2,:));
    [Xn,Yn] = meshgrid(Xcoord,Ycoord);

    %ÀûÓÃgriddataÀ´²åÖµ£¬´ÓxyzÉú³ÉÕ¤¸ñÊý¾Ý
    %×îºóÒ»¸öÎª²åÖµ·½·¨£¬°ülinear cubic natural nearestºÍv4µÈ·½·¨
    %v4·½·¨ºÄÊ±³¤µ«½ÏÎª×¼È·
    surf_new_solu_err = griddata(meshXY(1,:),meshXY(2,:),new_solu,Xn,Yn,'v4'); 
    surf(Xn,Yn,surf_new_solu_err);
    colorbar;
%     shading interp

%     figure('name', 'umesh')
%     mesh(Xn,Yn,surf_new_solu)
%     colorbar;
%     hold on
    %plot3(meshXY(1,:),meshXY(1,:),u,'r+','MarkerSize',3)
end