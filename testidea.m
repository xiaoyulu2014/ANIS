% space is (0,2] arm 1 is (0,1] and arm 2 is (1,2]
% target density
%truep = @(x) (x>1).*(x-1) + .0;
truep = @(x) exp(-10*x);
% construct discretized version
delta = .01;
xx = delta:delta:1;
allp = truep(xx);
allp = allp / sum(allp)/delta;
px{1} = allp(1:end/2);
px{2} = allp(end/2+1:end);
pp = [sum(px{1}) sum(px{2})]*delta; % target probs
draw = @(a) px{a}(ceil(rand(1)*length(px{a}))); % samples p

T = 10000;
sample = cell(1,2);
sample{1} = zeros(1,T);
sample{2} = zeros(1,T);

sample{1}(1) = draw(1);
sample{2}(1) = draw(2);
total = [sample{1}(1) sample{2}(1)];
dev = total.^2;
nn = ones(1,2);
cost = zeros(1,T+1);


for tt=1:T
  % estimation on normal hierarchical model (not sure if correct)
  mm = total./nn;
  vv = (dev./nn-mm.^2)./nn;
  mx = (total + fliplr(mm))./(nn+1);
  vx = ((dev+fliplr(mm.^2+vv))./(nn) - mx.^2)./(nn+1);
  % optimal qq minimizing E[p^2/q]-1 loss (reduce variance)
  qq = sqrt(vx+mx.^2);
  qq = qq / sum(qq);
  % cost
  cost(tt) = sum(pp.^2./qq)-1;
  % draw sample and update suff stats
  aa = 1+(rand(1)<qq(2));
  dd = draw(aa);
  total(aa) = total(aa) + dd;
  dev(aa)   = dev(aa)   + dd.^2;
  nn(aa)    = nn(aa)    + 1;
  sample{aa}(nn(aa)) = dd;
  % plot
  subplot(121); bar(1:2,[pp;qq;total./nn/sum(total./nn)]');
  subplot(122); semilogy(cost(1:tt));
  drawnow
  if cost(tt)<1e-12, break; end
end