function [ dwnData ] = parserEpanet( epanetInputFilename )
%
% pharserEpanet pharse the input EPANET file to obtain the topology 
%  of the network and stores into the dwnData structure. 
%
% SYNTAX :  dwnData = pharserEpanet( fileName )
% 
% INPUT  :  epanetInputFilename : Input EPANET file of the network under
%   consideration 
% OUTPUT :  dwnData  :  struture with the topology of the drinking water
%   network based on the input epanet file
%

inputFileId = fopen(epanetInputFilename);
epanetFile = textscan(inputFileId, '%s');
nSize = length(epanetFile{1});

% junctions, tanks, pipes, pumps and valves are important for network
idElements = zeros(5, 2);
for idFileElement = 1:nSize
    if( strcmp(epanetFile{1}(idFileElement), '[JUNCTIONS]'))
        idElements(1, 1) = idFileElement;
    end
    if( strcmp(epanetFile{1}(idFileElement), '[RESERVOIRS]') )
        idElements(1, 2) = idFileElement - 1;
    end
    if( strcmp(epanetFile{1}(idFileElement), '[TANKS]') )
        idElements(2, 1) = idFileElement;
    end
    if( strcmp(epanetFile{1}(idFileElement), '[PIPES]') )
        idElements(3, 1) = idFileElement;
        idElements(2, 2) = idFileElement - 1;
    end
    if( strcmp(epanetFile{1}(idFileElement), '[PUMPS]'))
        idElements(4, 1) = idFileElement;
        idElements(3, 2) = idFileElement - 1;
    end
    if( strcmp(epanetFile{1}(idFileElement), '[VALVES]') )
        idElements(5, 1) = idFileElement;
        idElements(4, 2) = idFileElement - 1;
    end
    if( strcmp( epanetFile{1}(idFileElement), '[TAGS]') )
        idElements(5, 2) = idFileElement - 1;
    end 
end 

% get infromation about the demands from the JUNCTION data 
junctionData = epanetFile{1}(idElements(1, 1):idElements(1, 2));
junctionDataLength = idElements(1, 2) - idElements(1, 1);
idFileElement = 0;
idCount = 1;
junctionId = [];
demand = [];
for junctionIter = 6:junctionDataLength
    junctionStringData = cell2mat(junctionData(junctionIter));
    if( idCount == 1)
        junctionId = [junctionId; junctionData(junctionIter)];
        idCount = idCount + 1;
    elseif( idCount == 3)
        demand = [demand; str2double(junctionData(junctionIter))];
        idCount = idCount + 1;
    else
        if( junctionStringData(end) == ';')
            idCount = 1;
        else 
            idCount = idCount + 1;
        end
    end
end
% get information about the tanks 
tankData = epanetFile{1}(idElements(2, 1):idElements(2, 2));
tankDataLength = idElements(2, 2) - idElements(2, 1);
idFileElement = 0;
idCount = 1;
tankId = [];
minTank = [];
maxTank = [];
initialLevel = [];
for tankIter = 10:tankDataLength
    tankStringData = cell2mat(tankData(tankIter));
    if( idCount == 1)
        tankId = [tankId; tankData(tankIter)];
        idCount = idCount + 1;
    elseif( idCount == 3)
        initialLevel = [initialLevel; str2double(tankData(tankIter))];
        idCount = idCount + 1;
    elseif( idCount == 4)
        minTank = [minTank; str2double(tankData(tankIter))];
        idCount = idCount + 1;
    elseif( idCount == 5)
        maxTank = [maxTank; str2double(tankData(tankIter))];
        idCount = idCount + 1;
    else
        if( tankStringData(end) == ';')
            idCount = 1;
        else 
            idCount = idCount + 1;
        end
    end
end
% get information about pipes 
pipeData = epanetFile{1}(idElements(3, 1):idElements(3, 2));
pipeDataLength = idElements(3, 2) - idElements(3, 1);
idFileElement = 0;
idCount = 1;
pipeId = [];
pipeNode1 = [];
pipeNode2 = [];
for pipeIter = 10:pipeDataLength
    pipeStringData = cell2mat(pipeData(pipeIter));
    if( idCount == 1)
        pipeId = [pipeId; pipeData(pipeIter)];
        idCount = idCount + 1;
    elseif( idCount == 2)
        pipeNode1 = [pipeNode1; pipeData(pipeIter)];
        idCount = idCount + 1;
    elseif( idCount == 3)
        pipeNode2 = [pipeNode2; pipeData(pipeIter)];
        idCount = idCount + 1;
    else
        if( pipeStringData(end) == ';')
            idCount = 1;
        else 
            idCount = idCount + 1;
        end
    end
end
% get information about pumps 
pumpData = epanetFile{1}(idElements(4, 1):idElements(4, 2));
pumpDataLength = idElements(4, 2) - idElements(4, 1);
idFileElement = 0;
idCount = 1;
pumpId = [];
pumpNode1 = [];
pumpNode2 = [];
for pumpIter = 6:pumpDataLength
    pumpStringData = cell2mat(pumpData(pumpIter));
    if( idCount == 1)
        pumpId = [pumpId; pumpData(pumpIter)];
        idCount = idCount + 1;
    elseif( idCount == 2)
        pumpNode1 = [pumpNode1; pumpData(pumpIter)];
        idCount = idCount + 1;
    elseif( idCount == 3)
        pumpNode2 = [pumpNode2; pumpData(pumpIter)];
        idCount = idCount + 1;
    else
        if( pumpStringData(end) == ';')
            idCount = 1;
        else 
            idCount = idCount + 1;
        end
    end
end
% get information about valves
valveData = epanetFile{1}(idElements(5, 1):idElements(5, 2));
valveDataLength = idElements(5, 2) - idElements(5, 1);
idFileElement = 0;
idCount = 1;
valveId = [];
valveNode1 = [];
valveNode2 = [];
for valveIter = 9:valveDataLength
    valveStringData = cell2mat(valveData(valveIter));
    if( idCount == 1)
        valveId = [valveId; valveData(valveIter)];
        idCount = idCount + 1;
    elseif( idCount == 2)
        valveNode1 = [valveNode1; valveData(valveIter)];
        idCount = idCount + 1;
    elseif( idCount == 3)
        valveNode2 = [valveNode2; valveData(valveIter)];
        idCount = idCount + 1;
    else
        if( valveStringData(end) == ';')
            idCount = 1;
        else 
            idCount = idCount + 1;
        end
    end
end
% generate the topology matrix
nTank = length(tankId);
nPump = length(pumpId);
nValve = length(valveId);
nInput = nPump + nValve;
nDemand = length(junctionId);
nPipe = length(pipeId);
A = eye(nTank);
Bd = zeros(nTank, nInput);
Gd = zeros(nTank, nDemand);
for iTank = 1:nTank
    currentTankId = tankId(iTank);
    for iPump = 1:nPump
        if( strcmp( pumpNode1(iPump), currentTankId) )
            Bd(iTank, iPump) = 1;
        elseif(strcmp( pumpNode2(iPump), currentTankId) )
            Bd(iTank, iPump) = -1;
        else
            Bd(iTank, iPump) = 0;
        end
    end
    for iValve = 1:nValve
        if( strcmp( valveNode1(iPump), currentTankId) )
            Bd(iTank, nPump + iValve) = 1;
        elseif(strcmp( pumpNode2(iPump), currentTankId) )
            Bd(iTank, nPump + iValve) = -1;
        else
            Bd(iTank, nPump + iValve) = 0;
        end
    end
    pipeConnectedTank = [];
    for iPipe = 1:nPipe
       if(strcmp(currentTankId, pipeNode1(iPipe)))
           pipeConnectedTank = [pipeConnectedTank; pipeNode2(iPipe)];
       elseif(strcmp(currentTankId, pipeNode2(iPipe)))
           pipeConnectedTank = [pipeConnectedTank; pipeNode1(iPipe)];
       end  
    end
    for iDemand = 1:nDemand
       for iPipeTank = 1:length(pipeConnectedTank)
           if( strcmp(pipeConnectedTank(iPipeTank), junctionId(iDemand)))
               Gd(iTank, iDemand) = 1;
           end 
       end
    end 
end 
% get input-demand coupling 
E =[];
Ed = [];
for iDemand = 1:nDemand
    currentDemandId = junctionId(iDemand);
    rowE = zeros(1, nInput);
    rowEd = zeros(1, nDemand);
    rowFlag = 0;
    for iPump = 1:nPump
        if(strcmp( currentDemandId, pumpNode1(iPump)))
            rowE(1, iPump) = 1;
            rowEd(1, iDemand) = -1;
            rowFlag = 1;
        end
        if(strcmp( currentDemandId, pumpNode2(iPump)))
            rowE(1, iPump) = -1;
            rowEd(1, iDemand) = -1;
            rowFlag = 1;
        end 
    end 
    for iValve = 1:nValve
        if(strcmp( currentDemandId, valveNode1))
            rowE(1, nPump + iValve) = 1;
            rowEd(1, iDemand) = -1;
            rowFlag = 1;
        end
        if(strcmp( currentDemandId, valveNode2))
            rowE(1, nPump + iPump) = -1;
            rowEd(1, iDemand) = -1;
            rowFlag = 1;
        end
    end
    if( rowFlag > 0 )
        E = [E; rowE];
        Ed = [Ed; rowEd];
    end 
end 

if isempty(E)
    E = zeros(1, nInput);
    Ed = zeros(1, nDemand);
end 

dwnData.nx = length(A);
dwnData.nu = size(Bd, 2);
dwnData.ne = size(E) * [1;0];
dwnData.nd = size(Gd, 2);
dwnData.matA = A;
dwnData.matB = Bd;
dwnData.matGd = Gd;
dwnData.matE = E;
dwnData.matEd = Ed;
dwnData.vecXmin = minTank;
dwnData.vecXmax = maxTank;
dwnData.vecUmin = zeros(dwnData.nu, 1);
dwnData.vecUmax = ones(dwnData.nu, 1) * 100;
dwnData.costAlpha1 = 10*ones(dwnData.nu, 1);

end

