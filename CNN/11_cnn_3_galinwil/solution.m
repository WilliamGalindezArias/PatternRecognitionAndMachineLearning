%--------------------------------------------------------------------
%% Init MatConvNet framework
% --------------------------------------------------------------------
MCNPath = '../matconvnet-1.0-beta23/matconvnet-1.0-beta23';
run(fullfile(MCNPath, 'matlab/vl_setupnn'))  

% load data
load imdb

% normalise the data
imdb.images.data = imdb.images.data / 255;

% training and validation sets
imdb.images.set = [1 * ones(1, 50000), 2 * ones(1, 10000), 3 * ones(1, 10000)];

%--------------------------------------------------------------------
%% 1. Softmax with SGD
% --------------------------------------------------------------------
delete expDir/*
clear net;

% build the network
net.layers = {};
net.layers{end+1} = struct('name', 'full', ...
			   'type', 'conv', ...
			   'weights', {{1e-2*randn(28,28,1,10,'single'), zeros(1, 10,'single')}}, ...
			   'stride', 1, ...
			   'pad', 0);
net.layers{end+1} = struct('type', 'softmaxloss');

net = vl_simplenn_tidy(net);
vl_simplenn_display(net)

[net, info] = cnn_train(net, imdb, @getSimpleNNBatch, 'batchSize', 1000, 'numEpochs', 99, 'expDir', 'expDir', 'plotStatistics', false); 
[net, info] = cnn_train(net, imdb, @getSimpleNNBatch, 'batchSize', 1000, 'numEpochs', 100, 'expDir', 'expDir');


% adding more layers + sigmoid/ReLU
delete expDir/*
clear net
net.layers = {};
net.layers{end+1} = struct('name', 'full1', ...
			   'type', 'conv', ...
			   'weights', {{1e-2*randn(28,28,1,1000,'single'), zeros(1, 1000,'single')}}, ...
			   'stride', 1, ...
			   'pad', 0);
net.layers{end+1} = struct('name', 'relu1', ...
			   'type', 'relu');
%net.layers{end+1} = struct('name', 'sigmoid1', ...
%			   'type', 'sigmoid');
net.layers{end+1} = struct('name', 'full2', ...
			   'type', 'conv', ...
			   'weights', {{1e-2*randn(1,1,1000,100,'single'), zeros(1, 100,'single')}}, ...
			   'stride', 1, ...
			   'pad', 0);
net.layers{end+1} = struct('name', 'relu1', ...
			   'type', 'relu');
net.layers{end+1} = struct('name', 'conv3', ...
			   'type', 'conv', ...
			   'weights', {{1e-2*randn(1,1,100,10,'single'), zeros(1, 10,'single')}}, ...
			   'stride', 1, ...
			   'pad', 0);
%net.layers{end+1} = struct('name', 'conv4', ...
%			   'type', 'conv', ...
%			   'weights', {{1e-2*randn(1,1,1000,10,'single'), zeros(1, 10,'single')}}, ...
%			   'stride', 1, ...
%			   'pad', 0);
net.layers{end+1} = struct('type', 'softmaxloss');

net = vl_simplenn_tidy(net);
vl_simplenn_display(net)

[net, info] = cnn_train(net, imdb, @getSimpleNNBatch, 'batchSize', 1000, 'numEpochs', 99, 'expDir', 'expDir', 'plotStatistics', false); 
[net, info] = cnn_train(net, imdb, @getSimpleNNBatch, 'batchSize', 1000, 'numEpochs', 100, 'expDir', 'expDir');


%--------------------------------------------------------------------
%% 2. CNNs
% --------------------------------------------------------------------

clear net;
delete expDir/*

net.layers = {} ;
net.layers{end+1} = struct('name', 'conv1', ...
			   'type', 'conv', ...
			   'weights', {{1e-2*randn(3,3,1,10,'single'), zeros(1, 10,'single')}}, ...
			   'stride', 2, ...
			   'pad', 1) ;
net.layers{end+1} = struct('name', 'relu1', ...
			   'type', 'relu') ;
net.layers{end+1} = struct('name', 'full', ...
			   'type', 'conv', ...
			   'weights', {{1e-2*randn(14,14,10,10,'single'), zeros(1, 10,'single')}}, ... %randn(1,10,'single')}}, 
			   'stride', 1, ...
			   'pad', 0);
net.layers{end+1} = struct('type', 'softmaxloss') ;

net = vl_simplenn_tidy(net);
vl_simplenn_display(net)

[net, info] = cnn_train(net, imdb, @getSimpleNNBatch, 'batchSize', 1000, 'numEpochs', 99, 'expDir', 'expDir', 'plotStatistics', false); 
[net, info] = cnn_train(net, imdb, @getSimpleNNBatch, 'batchSize', 1000, 'numEpochs', 100, 'expDir', 'expDir');


% adding layers, dropout and max-pooling
clear net;
delete expDir/*

net.layers = {} ;
net.layers{end+1} = struct('name', 'conv1', ...
			   'type', 'conv', ...
			   'weights', {{1e-2*randn(3,3,1,10,'single'), zeros(1, 10,'single')}}, ...
			   'stride', 1, ...
			   'pad', 1) ;
net.layers{end+1} = struct('name', 'relu1', ...
			   'type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('name', 'conv2', ...
			   'type', 'conv', ...
			   'weights', {{1e-2*randn(3,3,10,10,'single'), zeros(1, 10,'single')}}, ...
			   'stride', 1, ...
			   'pad', 1) ;
net.layers{end+1} = struct('name', 'relu2', ...
			   'type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5) ;
net.layers{end+1} = struct('name', 'full', ...
			   'type', 'conv', ...
			   'weights', {{1e-2*randn(7,7,10,10,'single'), zeros(1, 10,'single')}}, ... %randn(1,10,'single')}}, 
			   'stride', 1, ...
			   'pad', 0);
net.layers{end+1} = struct('type', 'softmaxloss') ;

net = vl_simplenn_tidy(net);
vl_simplenn_display(net)

[net, info] = cnn_train(net, imdb, @getSimpleNNBatch, 'batchSize', 1000, 'numEpochs', 199, 'expDir', 'expDir', 'plotStatistics', false); 
[net, info] = cnn_train(net, imdb, @getSimpleNNBatch, 'batchSize', 1000, 'numEpochs', 200, 'expDir', 'expDir');



%% Helper functions

% dropout layer
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5) ;
% max-pooling layer
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

% save the best network
net.layers{end}.type = 'softmax';
save('my_net.mat', 'net');


%% 3. Test functions
load imdb_test.mat
imdb_test.images.data = imdb_test.images.data / 255;

res = vl_simplenn(net, imdb_test.images.data, [], [], 'Mode', 'test');
y = squeeze(res(end).x);
[~, L] = max(y);
test_err = sum(L ~= imdb_test.images.labels) / length(imdb_test.images.labels);


% --------------------------------------------------------------------
%% Visualize results
% --------------------------------------------------------------------

k = 100;

figure(1);
print('-dpng',['output/learning_error',sprintf('%03d',k),'.png']);

figure(2); clf;
% take first testing image and its label
idx = 60006;
im = imdb.images.data(:,:,1,idx);
label = imdb.images.labels(idx);
imagesc(im + imdb.images.data_mean);
colormap(gray); 
print('-dpng',['output/input_',sprintf('%03d',k),'.png']);
% show corresponding weight matrix
figure(3); clf;
%imagesc(net.layers{end-1}.weights{1}(:,:,1,label)); colorbar
for a = 1:10, subplot(1,10,a); imagesc(squeeze(net.layers{2}.weights{1}(:,:,1,a))); end;
colormap(gray);
print('-dpng',['output/weights_',sprintf('%03d',k),'.png']);

% add corrrect label to the softmax layer
net.layers{end}.class = label;
% compute response
res = vl_simplenn(net, im, [], [], 'Mode', 'test');

figure(4); clf;
bar(0:9, squeeze(res(end-1).x(:,:,1:10,1)));
print('-dpng',['output/output_',sprintf('%03d',k),'.png']);
