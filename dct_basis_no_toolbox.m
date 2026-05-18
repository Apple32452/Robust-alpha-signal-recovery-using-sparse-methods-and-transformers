function C = dct_basis_no_toolbox(N)
C = zeros(N,N);
for k = 0:N-1
    if k == 0, alpha = sqrt(1/N); else, alpha = sqrt(2/N); end
    for n = 0:N-1
        C(k+1, n+1) = alpha * cos(pi*(2*n+1)*k/(2*N));
    end
end
end
