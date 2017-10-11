function generateJsonFile( dataStruct, pathToFile )
% 
% Open a file with the spesified title and generate the json file from 
% the structure 
% 
% Syntax 
% Input     
%      dataStruct  : Contains the data that need to be converted to
%                    the json file 
%      pathToFile  : string that contain the path to the file 
%

fieldNamesData = fieldnames(dataStruct);
f = fopen(pathToFile, 'w+');
fprintf(f,'%s \n','{');
for i = 1:length(fieldNamesData)
    currentData = dataStruct.(fieldNamesData{i});
    if isscalar(currentData)
        fprintf(f, '"%s" : [', fieldNamesData{i});
        %currentDim = size(currentData);
        fprintf(f, '%d]', currentData);
    else
        if(ischar(currentData))
            fprintf(f, '"%s" : ', fieldNamesData{i});
            fprintf(f,'"%s"', currentData);
        else
            fprintf(f, '"%s" : [', fieldNamesData{i});
            currentDim = size(currentData);
            for jj = 1 : currentDim(2)
                for kk = 1 : currentDim(1)
                    if (jj*kk < numel(currentData))
                        fprintf(f, '%d, ', currentData( kk, jj));
                    else
                        fprintf(f, '%d]', currentData(kk, jj));
                    end
                end
                if (jj*kk < numel(currentData))
                    fprintf(f, '\n');
                end
            end
        end
    end
    
    if i< length(fieldNamesData)
        fprintf(f, ',\n');
    end
end
fprintf(f,'\n%s','}');


end

