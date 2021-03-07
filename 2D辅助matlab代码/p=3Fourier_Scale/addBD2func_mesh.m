 function u_solu = addBD2func_mesh(mesh_points, solu)
  if (size(solu,1)~=1)
      solu = solu';
  end
  dim2var = size(mesh_points,1);
  num2points = size(mesh_points,2);
  if size(solu,1)~=num2points
      u_solu = zeros(1,num2points);
      js=1;
      for i=1:1:num2points
          if (mesh_points(1,i)==0 || mesh_points(1,i)==1 || mesh_points(2,i)==0 || mesh_points(2,i)==1)
              u_solu(1,i)=0;
          else
              if js<=size(solu,2)
                u_solu(1,i)=solu(1,js);
                js=js+1;
              else
                 u_solu(1,i) =0;
              end
          end
      end
  else
      u_solu = solu;
  end
end