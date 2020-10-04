load("sk.mat")
for dim=[64, 128]
for lambda=[1e-3, 1e-2, 0.1, 0.5]
str_e = sprintf("sk-%d-%0.5f.mat", dim, lambda)
[U, V] = sql_fresh(tr, te, dim, lambda);
save(str_e, 'U', 'V');
end
end