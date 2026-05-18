function y = soft_threshold(x, tau)
y = sign(x) .* max(abs(x) - tau, 0);
end
