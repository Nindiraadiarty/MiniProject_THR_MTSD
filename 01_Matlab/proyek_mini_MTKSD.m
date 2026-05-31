%% =========================================================
%  PROYEK MINI SAINS DATA  Pertemuan 12
%  Tema 3: BMI vs Tekanan Darah Sistolik
%  Dataset: Cardiovascular Disease (sulianova - Kaggle)
%% =========================================================

clc; clear; close all;

%% -----------------------------------------------------------
%  STEP 1: MUAT DATA
%% -----------------------------------------------------------
opts = detectImportOptions('cardio_train.csv', 'Delimiter', ';');
T    = readtable('cardio_train.csv', opts);

disp('=== Nama Kolom Dataset ===');
disp(T.Properties.VariableNames);
disp('=== 5 Baris Pertama ===');
disp(head(T, 5));
fprintf('Jumlah data awal: %d baris\n', height(T));

%% -----------------------------------------------------------
%  STEP 2: HITUNG BMI
%  BMI = weight (kg) / (height (m))^2
%  height dalam cm -> dibagi 100
%% -----------------------------------------------------------
height_m = T.height / 100;
BMI_raw  = T.weight ./ (height_m .^ 2);
SBP_raw  = T.ap_hi;

fprintf('\n--- Sebelum Cleaning ---\n');
fprintf('BMI  | Min: %6.1f | Max: %6.1f | Mean: %6.1f\n', min(BMI_raw), max(BMI_raw), mean(BMI_raw));
fprintf('SBP  | Min: %6.1f | Max: %6.1f | Mean: %6.1f\n', min(SBP_raw), max(SBP_raw), mean(SBP_raw));

%% -----------------------------------------------------------
%  STEP 3: DATA CLEANING
%  BMI  : 10 - 70  kg/m²  (rentang valid manusia dewasa)
%  SBP  : 50 - 250 mmHg   (di luar rentang = error input)
%% -----------------------------------------------------------
idx_valid = (BMI_raw >= 10) & (BMI_raw <= 70) & ...
            (SBP_raw >= 50) & (SBP_raw <= 250);

BMI = BMI_raw(idx_valid);
SBP = SBP_raw(idx_valid);

fprintf('\n--- Sesudah Cleaning ---\n');
fprintf('Data valid  : %d baris\n', length(BMI));
fprintf('Data dihapus: %d baris\n', length(BMI_raw) - length(BMI));

%% -----------------------------------------------------------
%  STEP 4: STATISTIK DESKRIPTIF
%% -----------------------------------------------------------
fprintf('\n=== Statistik Deskriptif ===\n');
fprintf('%-10s %8s %8s %8s %8s %8s\n', 'Variabel','Min','Max','Mean','Median','Std');
fprintf('%-10s %8.2f %8.2f %8.2f %8.2f %8.2f\n', 'BMI', min(BMI), max(BMI), mean(BMI), median(BMI), std(BMI));
fprintf('%-10s %8.2f %8.2f %8.2f %8.2f %8.2f\n', 'SBP', min(SBP), max(SBP), mean(SBP), median(SBP), std(SBP));

%% -----------------------------------------------------------
%  STEP 5: VISUALISASI EDA
%% -----------------------------------------------------------

% Figure 1: Histogram BMI
figure(1);
histogram(BMI, 40, 'FaceColor', '#4472C4', 'EdgeColor', 'white');
xlabel('BMI (kg/m²)'); ylabel('Frekuensi');
title('Distribusi BMI Pasien');
grid on;

% Figure 2: Histogram SBP
figure(2);
histogram(SBP, 40, 'FaceColor', '#ED7D31', 'EdgeColor', 'white');
xlabel('Tekanan Darah Sistolik (mmHg)'); ylabel('Frekuensi');
title('Distribusi Tekanan Darah Sistolik');
grid on;

% Figure 3: Scatter Plot
figure(3);
scatter(BMI, SBP, 8, 'filled', 'MarkerFaceColor', '#4472C4', 'MarkerFaceAlpha', 0.15);
xlabel('BMI (kg/m²)'); ylabel('Tekanan Darah Sistolik (mmHg)');
title('Scatter Plot: BMI vs Tekanan Darah Sistolik');
grid on;

% Figure 4: Boxplot sebelum & sesudah cleaning
figure(4);
subplot(1,2,1);
boxplot(BMI_raw, 'Labels', {'BMI (sebelum)'}); title('BMI Sebelum Cleaning'); grid on;
subplot(1,2,2);
boxplot(BMI, 'Labels', {'BMI (sesudah)'}); title('BMI Sesudah Cleaning'); grid on;

%% -----------------------------------------------------------
%  STEP 6: REGRESI LINIER & PARABOLIK
%% -----------------------------------------------------------
X = BMI;
Y = SBP;

% Model Linier: y = b1*x + b0
p1      = polyfit(X, Y, 1);
Y_pred1 = polyval(p1, X);

fprintf('\n=== Model Regresi Linier ===\n');
fprintf('y = %.4f*x + %.4f\n', p1(1), p1(2));

% Model Parabolik: y = c*x^2 + b*x + a
p2      = polyfit(X, Y, 2);
Y_pred2 = polyval(p2, X);

fprintf('\n=== Model Regresi Parabolik ===\n');
fprintf('y = %.4f*x^2 + %.4f*x + %.4f\n', p2(1), p2(2), p2(3));

% Figure 5: Plot perbandingan model
X_plot = linspace(min(X), max(X), 300);
figure(5);
scatter(X, Y, 6, 'filled', 'MarkerFaceColor', '#4472C4', 'MarkerFaceAlpha', 0.1, 'DisplayName', 'Data Pasien');
hold on;
plot(X_plot, polyval(p1, X_plot), 'r-',  'LineWidth', 2.5, 'DisplayName', 'Regresi Linier');
plot(X_plot, polyval(p2, X_plot), 'g-',  'LineWidth', 2.5, 'DisplayName', 'Regresi Parabolik');
xlabel('BMI (kg/m²)'); ylabel('Tekanan Darah Sistolik (mmHg)');
title('Regresi Linier vs Parabolik: BMI vs SBP');
legend('Location', 'best'); grid on; hold off;

%% -----------------------------------------------------------
%  STEP 7: EVALUASI MODEL (R², RMSE, MAE) - tanpa function
%% -----------------------------------------------------------
SS_tot  = sum((Y - mean(Y)).^2);

SS_res1 = sum((Y - Y_pred1).^2);
R2_1    = 1 - SS_res1/SS_tot;
RMSE_1  = sqrt(mean((Y - Y_pred1).^2));
MAE_1   = mean(abs(Y - Y_pred1));

SS_res2 = sum((Y - Y_pred2).^2);
R2_2    = 1 - SS_res2/SS_tot;
RMSE_2  = sqrt(mean((Y - Y_pred2).^2));
MAE_2   = mean(abs(Y - Y_pred2));

fprintf('\n=== Evaluasi Model ===\n');
fprintf('%-12s  R2 = %.4f  |  RMSE = %.4f  |  MAE = %.4f\n', 'Linier',    R2_1, RMSE_1, MAE_1);
fprintf('%-12s  R2 = %.4f  |  RMSE = %.4f  |  MAE = %.4f\n', 'Parabolik', R2_2, RMSE_2, MAE_2);

if R2_2 > R2_1
    fprintf('>> Model PARABOLIK lebih baik (R2 lebih tinggi)\n');
else
    fprintf('>> Model LINIER lebih baik (R2 lebih tinggi)\n');
end

% Figure 6: Residual Plot
res1 = Y - Y_pred1;
res2 = Y - Y_pred2;

figure(6);
subplot(1,2,1);
scatter(Y_pred1, res1, 5, 'filled', 'MarkerFaceColor','#ED7D31','MarkerFaceAlpha',0.1);
yline(0,'k--','LineWidth',1.5);
xlabel('Nilai Fitted'); ylabel('Residual');
title('Residual Plot - Linier'); grid on;

subplot(1,2,2);
scatter(Y_pred2, res2, 5, 'filled', 'MarkerFaceColor','#70AD47','MarkerFaceAlpha',0.1);
yline(0,'k--','LineWidth',1.5);
xlabel('Nilai Fitted'); ylabel('Residual');
title('Residual Plot - Parabolik'); grid on;

%% -----------------------------------------------------------
%  STEP 8: PREDIKSI DATA BARU
%% -----------------------------------------------------------
BMI_test = [16.0, 18.5, 22.0, 25.0, 27.5, 30.0, 35.0, 40.0];
kategori = {'Sangat Kurus','Kurus','Normal','Normal Atas', ...
            'Overweight','Obesitas I','Obesitas II','Obesitas III'};

fprintf('\n=== Prediksi SBP dari Model Parabolik ===\n');
fprintf('%-15s  %-22s  %-20s\n', 'BMI (kg/m2)', 'SBP Prediksi (mmHg)', 'Kategori BMI');
fprintf('%s\n', repmat('-',1,60));
for i = 1:length(BMI_test)
    fprintf('%-15.1f  %-22.2f  %s\n', BMI_test(i), polyval(p2, BMI_test(i)), kategori{i});
end

% Figure 7: Plot prediksi
figure(7);
scatter(X, Y, 6, 'filled', 'MarkerFaceColor','#4472C4','MarkerFaceAlpha',0.1,'DisplayName','Data Pasien');
hold on;
plot(X_plot, polyval(p2, X_plot), 'g-', 'LineWidth', 2.5, 'DisplayName','Model Parabolik');
scatter(BMI_test, polyval(p2, BMI_test), 100, 'r^', 'filled', 'DisplayName','Titik Prediksi');
xlabel('BMI (kg/m²)'); ylabel('Tekanan Darah Sistolik (mmHg)');
title('Prediksi SBP Berdasarkan Model Parabolik');
legend('Location','best'); grid on; hold off;

%% -----------------------------------------------------------
%  STEP 9: PERHITUNGAN MANUAL REGRESI LINIER (Tabel Sigma)
%% -----------------------------------------------------------
n   = length(X);
Sx  = sum(X);
Sy  = sum(Y);
Sx2 = sum(X.^2);
Sxy = sum(X.*Y);

fprintf('\n=== Perhitungan Manual Regresi Linier (Tabel Sigma) ===\n');
fprintf('n       = %d\n',   n);
fprintf('SigmaX  = %.4f\n', Sx);
fprintf('SigmaY  = %.4f\n', Sy);
fprintf('SigmaX2 = %.4f\n', Sx2);
fprintf('SigmaXY = %.4f\n', Sxy);

A    = [n, Sx; Sx, Sx2];
B    = [Sy; Sxy];
koef = A \ B;

fprintf('\nHasil Persamaan Normal:\n');
fprintf('b0 (intercept) = %.4f\n', koef(1));
fprintf('b1 (slope)     = %.4f\n', koef(2));
fprintf('\nVerifikasi vs polyfit:\n');
fprintf('polyfit -> b0 = %.4f | b1 = %.4f\n', p1(2), p1(1));
fprintf('(Kedua nilai identik = perhitungan benar)\n');

fprintf('\n========================================\n');
fprintf('SEMUA LANGKAH SELESAI\n');
fprintf('========================================\n');
