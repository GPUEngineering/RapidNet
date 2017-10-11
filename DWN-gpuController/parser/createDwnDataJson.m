%%
clc
clear
% Path to the json files
pathToNetwork = 'network.json';
epanetInputFile = 'testEpanet.inp';
dwnData = pharserEpanet( epanetInputFile );
%load('dwn');
P.Hp = 24;
P.Hu = 23;
P.xs = 0.35*dwnData.xmax;
% Network model and the creating the json file for this model.
DWNnetwork.nx = length(dwnData.A);
DWNnetwork.nu = size(dwnData.B, 2);
DWNnetwork.ne = size(dwnData.E) * [1;0];
DWNnetwork.nd = size(dwnData.Gd, 2);
DWNnetwork.N = P.Hp;
DWNnetwork.matA = dwnData.A;
DWNnetwork.matB = dwnData.B;
DWNnetwork.matGd = dwnData.Gd;
DWNnetwork.matE = dwnData.E;
DWNnetwork.matEd = dwnData.Ed;
DWNnetwork.vecXmin = dwnData.xmin;
DWNnetwork.vecXmax = dwnData.xmax;
DWNnetwork.vecXsafe = P.xs;
DWNnetwork.vecUmin = zeros(DWNnetwork.nu, 1);
DWNnetwork.vecUmax = ones(DWNnetwork.nu, 1) * 100;
DWNnetwork.costAlpha1 = 10*ones(DWNnetwork.nu, 1);

generateJsonFile( DWNnetwork, pathToNetwork);