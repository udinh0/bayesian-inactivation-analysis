# ============================================
# Modelo GLM Bayesiano para Inativação Térmica
# Bacillus simplex - Código Original do Artigo
# ============================================

# Pacotes necessários
library(rstan)
library(tidyverse)
library(bayesplot)

# Configuração do Stan
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
set.seed(175072)

# ============================================
# 1. CARREGAR DADOS REAIS
# ============================================

# Dados de treinamento (10^5 células)
dados_treino <- read_csv("data/bayesian_fitting.csv")

# Dados de validação
dados_850 <- read_csv("data/850cell.csv")
dados_90 <- read_csv("data/90cell.csv")
dados_8 <- read_csv("data/8cell.csv")

# Visualizar estrutura
cat("=== DADOS DE TREINAMENTO ===\n")
print(dados_treino)
cat("\nObservações iniciais (Initial != 0):", sum(dados_treino$Initial != 0), "\n")
cat("Observações de inativação (Initial == 0):", sum(dados_treino$Initial == 0), "\n")

# ============================================
# 2. MODELO STAN ORIGINAL DO ARTIGO
# ============================================

stan_code <- "
data {
  int M;
  int Nt[M];
  real TIME[M];
  int Initial[M];
}

parameters {
  real<lower=0> a;      // delta (scale parameter)
  real<lower=0> b;      // p (shape parameter)
  real<lower=0> n0;     // concentração inicial
}

model {
  // Priors (uniforme)
  
  // Likelihood para medições iniciais (t=0)
  for (m in 1:M) {
    if (Initial[m] != 0) {
      Nt[Initial[m]] ~ poisson(n0);
    }
  }
  
  // Likelihood para todas as observações (modelo de inativação)
  for (m in 1:M) {        
    Nt[m] ~ poisson(n0 * (10^(-((TIME[m]/a)^b))));
  }
}

generated quantities {
  vector[M] log_lik;
  int y_pred[M];
  
  for (m in 1:M) {
    real lambda;
    lambda = n0 * (10^(-((TIME[m]/a)^b)));
    log_lik[m] = poisson_lpmf(Nt[m] | lambda);
    y_pred[m] = poisson_rng(lambda);
  }
}
"

# ============================================
# 3. PREPARAR DADOS PARA STAN
# ============================================

stan_data <- list(
  M = nrow(dados_treino),
  Nt = as.integer(dados_treino$Nt),
  TIME = dados_treino$TIME,
  Initial = as.integer(dados_treino$Initial)
)

cat("\n=== DADOS PARA STAN ===\n")
cat("M (número de observações):", stan_data$M, "\n")
cat("Range de Nt:", min(stan_data$Nt), "-", max(stan_data$Nt), "\n")
cat("Tempos únicos:", unique(stan_data$TIME), "\n")

# ============================================
# 4. AJUSTE DO MODELO BAYESIANO
# ============================================

cat("\n🔄 Iniciando ajuste do modelo (pode demorar alguns minutos)...\n\n")

fit <- stan(
  model_code = stan_code,
  data = stan_data,
  chains = 4,
  iter = 10000,
  warmup = 5000,
  thin = 1,
  seed = 123,
  control = list(adapt_delta = 0.95, max_treedepth = 15)
)

cat("\n✓ Modelo ajustado com sucesso!\n\n")

# ============================================
# 5. DIAGNÓSTICOS DE CONVERGÊNCIA
# ============================================

# Sumário dos parâmetros (a=delta, b=p, n0=N0)
print(fit, pars = c("a", "b", "n0"))

# Extrair sumário
summary_fit <- summary(fit, pars = c("a", "b", "n0"))$summary
print(summary_fit)

# Criar diretório de output
dir.create("output", showWarnings = FALSE)

# Trace plots
png("output/trace_plots.png", width = 1200, height = 800)
mcmc_trace(fit, pars = c("a", "b", "n0"))
dev.off()

# Distribuições posteriores
png("output/posterior_distributions.png", width = 1200, height = 800)
mcmc_dens(fit, pars = c("a", "b", "n0"))
dev.off()

# Correlação entre parâmetros
png("output/parameter_pairs.png", width = 800, height = 800)
mcmc_pairs(fit, pars = c("a", "b"))
dev.off()

cat("✓ Gráficos de diagnóstico salvos em 'output/'\n")

# ============================================
# 6. EXTRAIR PARÂMETROS ESTIMADOS
# ============================================

posterior_samples <- rstan::extract(fit)

# Renomear para nomenclatura do artigo (a=delta, b=p)
delta_samples <- posterior_samples$a
p_samples <- posterior_samples$b
n0_samples <- posterior_samples$n0

cat("\n📊 PARÂMETROS ESTIMADOS:\n")
cat("delta (a): média =", round(mean(delta_samples), 2), 
    ", SD =", round(sd(delta_samples), 3), "\n")
cat("p (b): média =", round(mean(p_samples), 2), 
    ", SD =", round(sd(p_samples), 3), "\n")
cat("N0 (n0): média =", round(mean(n0_samples), 0), 
    ", SD =", round(sd(n0_samples), 0), "\n")
cat("Correlação (delta, p):", round(cor(delta_samples, p_samples), 3), "\n\n")

# ============================================
# 7. VISUALIZAR AJUSTE NO DATASET DE TREINAMENTO
# ============================================

# Calcular predições para os dados de treinamento
pred_treino <- apply(posterior_samples$y_pred, 2, function(x) {
  c(lower = quantile(x, 0.025),
    median = quantile(x, 0.5),
    upper = quantile(x, 0.975))
})

dados_treino_pred <- dados_treino %>%
  mutate(
    pred_lower = pred_treino[1, ],
    pred_median = pred_treino[2, ],
    pred_upper = pred_treino[3, ]
  )

# Plot do ajuste
p_fit <- ggplot(dados_treino_pred, aes(x = TIME)) +
  geom_ribbon(aes(ymin = pred_lower, ymax = pred_upper),
              alpha = 0.2, fill = "steelblue") +
  geom_line(aes(y = pred_median), color = "steelblue", size = 1.2) +
  geom_point(aes(y = Nt, color = factor(Initial)), size = 3) +
  scale_color_manual(values = c("black", "red", "green", "blue"),
                     labels = c("Inativação", "Rep 1", "Rep 2", "Rep 3"),
                     name = "Tipo") +
  scale_y_log10() +
  labs(title = "Ajuste do Modelo - Dataset de Treinamento",
       subtitle = "Dados: 10^5 células, 3 replicatas",
       x = "Tempo (min)",
       y = "Número de células (log10)") +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p_fit)
ggsave("output/ajuste_treinamento.png", p_fit, width = 10, height = 6, dpi = 300)

# ============================================
# 8. FUNÇÃO PARA SIMULAÇÃO DE VALIDAÇÃO
# ============================================

simular_inactivacao <- function(N0_inicial, tempos, n_sim = 20000) {
  
  # Usar amostras posteriores
  delta_sim <- delta_samples[1:n_sim]
  p_sim <- p_samples[1:n_sim]
  
  # Matriz para armazenar simulações
  resultados <- matrix(NA, nrow = n_sim, ncol = length(tempos))
  
  for (i in 1:n_sim) {
    delta_i <- delta_sim[i]
    p_i <- p_sim[i]
    
    # Gerar número inicial de células (Poisson)
    N0_real <- rpois(1, N0_inicial)
    
    # Simular sobrevivência em cada tempo
    for (j in 1:length(tempos)) {
      t_j <- tempos[j]
      lambda <- N0_real * 10^(-((t_j/delta_i)^p_i))
      resultados[i, j] <- rpois(1, max(0, lambda))
    }
  }
  
  return(resultados)
}

# ============================================
# 9. VALIDAÇÃO COM DIFERENTES CONCENTRAÇÕES
# ============================================

cat("🔄 Realizando simulações para validação...\n")

# Tempos únicos de cada dataset
tempos_850 <- sort(unique(dados_850$TIME))
tempos_90 <- sort(unique(dados_90$TIME))
tempos_8 <- sort(unique(dados_8$TIME))

# Simulações
sim_850 <- simular_inactivacao(N0_inicial = 850, tempos = tempos_850)
sim_90 <- simular_inactivacao(N0_inicial = 90, tempos = tempos_90)
sim_8 <- simular_inactivacao(N0_inicial = 8, tempos = tempos_8)

cat("✓ Simulações concluídas!\n\n")

# ============================================
# 10. CALCULAR INTERVALOS DE PREDIÇÃO
# ============================================

calcular_intervalos <- function(simulacoes, tempos) {
  intervalos <- apply(simulacoes, 2, function(x) {
    c(quantile(x, 0.025),
      quantile(x, 0.5),
      quantile(x, 0.975),
      mean(x))
  })
  
  df <- as.data.frame(t(intervalos))
  
  # NOMEAR AS COLUNAS EXPLICITAMENTE
  colnames(df) <- c("lower", "median", "upper", "mean")
  df$tempo <- tempos
  
  return(df)
}

int_850 <- calcular_intervalos(sim_850, tempos_850)
int_90 <- calcular_intervalos(sim_90, tempos_90)
int_8 <- calcular_intervalos(sim_8, tempos_8)

# ============================================
# 11. CALCULAR ACURÁCIA
# ============================================

calcular_acuracia <- function(dados_obs, intervalos) {
  
  dados_completos <- dados_obs %>%
    left_join(intervalos, by = c("TIME" = "tempo"))
  
  dentro <- (dados_completos$N >= dados_completos$lower) & 
    (dados_completos$N <= dados_completos$upper)
  
  acuracia <- sum(dentro, na.rm = TRUE) / nrow(dados_completos) * 100
  
  return(list(
    acuracia = acuracia,
    n_total = nrow(dados_completos),
    n_dentro = sum(dentro, na.rm = TRUE)
  ))
}

cat("\n📈 ACURÁCIA DAS PREDIÇÕES (intervalo de 95%):\n")

acc_850 <- calcular_acuracia(dados_850, int_850)
cat("850 células:", round(acc_850$acuracia, 1), "% ",
    "(", acc_850$n_dentro, "/", acc_850$n_total, ")\n")

acc_90 <- calcular_acuracia(dados_90, int_90)
cat("90 células:", round(acc_90$acuracia, 1), "% ",
    "(", acc_90$n_dentro, "/", acc_90$n_total, ")\n")

acc_8 <- calcular_acuracia(dados_8, int_8)
cat("8 células:", round(acc_8$acuracia, 1), "% ",
    "(", acc_8$n_dentro, "/", acc_8$n_total, ")\n\n")

# ============================================
# 12. VISUALIZAÇÃO DOS RESULTADOS
# ============================================

plot_validacao <- function(dados_obs, intervalos, n0_titulo) {
  
  # Dados agregados
  dados_obs_media <- dados_obs %>%
    group_by(TIME) %>%
    summarise(
      media = mean(N),
      sd = sd(N),
      n = n()
    )
  
  # Plot
  p <- ggplot() +
    geom_ribbon(data = intervalos, 
                aes(x = tempo, ymin = lower, ymax = upper),
                alpha = 0.2, fill = "steelblue") +
    geom_line(data = intervalos,
              aes(x = tempo, y = median),
              color = "steelblue", size = 1.2, linetype = "dashed") +
    geom_jitter(data = dados_obs,
                aes(x = TIME, y = N),
                alpha = 0.3, color = "gray40", size = 1.5, width = 0.05) +
    geom_point(data = dados_obs_media,
               aes(x = TIME, y = media),
               color = "red", size = 3) +
    geom_errorbar(data = dados_obs_media,
                  aes(x = TIME, ymin = pmax(media - sd, 0.1), ymax = media + sd),
                  color = "red", width = 0.1, size = 0.8) +
    scale_y_log10(limits = c(0.5, NA)) +
    annotation_logticks(sides = "l") +
    labs(title = paste0("Validação Estocástica - N0 ≈ ", n0_titulo, " células"),
         subtitle = "Pontos cinza = 60 replicatas | Vermelho = média ± SD | Azul = predição mediana e IC 95%",
         x = "Tempo de inativação (min)",
         y = "Número de células sobreviventes (log10)") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", size = 14),
          plot.subtitle = element_text(size = 9),
          panel.grid.minor = element_blank())
  
  return(p)
}

# Gerar plots
p1 <- plot_validacao(dados_850, int_850, "850")
p2 <- plot_validacao(dados_90, int_90, "90")
p3 <- plot_validacao(dados_8, int_8, "8")

# Salvar plots
ggsave("output/validacao_850cell.png", p1, width = 10, height = 6, dpi = 300)
ggsave("output/validacao_90cell.png", p2, width = 10, height = 6, dpi = 300)
ggsave("output/validacao_8cell.png", p3, width = 10, height = 6, dpi = 300)

print(p1)
print(p2)
print(p3)

cat("\n✓ Gráficos de validação salvos!\n")

# ============================================
# 13. SALVAR RESULTADOS
# ============================================

# Parâmetros estimados
parametros_estimados <- tibble(
  parametro = c("delta (a)", "p (b)", "N0 (n0)"),
  media = c(mean(delta_samples), mean(p_samples), mean(n0_samples)),
  sd = c(sd(delta_samples), sd(p_samples), sd(n0_samples)),
  q025 = c(quantile(delta_samples, 0.025),
           quantile(p_samples, 0.025),
           quantile(n0_samples, 0.025)),
  q975 = c(quantile(delta_samples, 0.975),
           quantile(p_samples, 0.975),
           quantile(n0_samples, 0.975))
)

write_csv(parametros_estimados, "output/parametros_estimados.csv")
cat("✓ Parâmetros salvos: output/parametros_estimados.csv\n")

# Acurácia
acuracias <- tibble(
  concentracao = c("850 células (10^3)", "90 células (10^2)", "8 células (10^1)"),
  acuracia_pct = c(acc_850$acuracia, acc_90$acuracia, acc_8$acuracia),
  n_dentro = c(acc_850$n_dentro, acc_90$n_dentro, acc_8$n_dentro),
  n_total = c(acc_850$n_total, acc_90$n_total, acc_8$n_total)
)

write_csv(acuracias, "output/acuracias_validacao.csv")
cat("✓ Acurácias salvas: output/acuracias_validacao.csv\n")

# Intervalos de predição
write_csv(int_850, "output/intervalos_predicao_850.csv")
write_csv(int_90, "output/intervalos_predicao_90.csv")
write_csv(int_8, "output/intervalos_predicao_8.csv")
cat("✓ Intervalos de predição salvos\n")

# Salvar workspace completo
saveRDS(list(
  fit = fit,
  posterior_samples = list(delta = delta_samples, p = p_samples, n0 = n0_samples),
  acuracias = acuracias,
  parametros = parametros_estimados
), "output/modelo_completo.rds")

cat("\n✅ ANÁLISE COMPLETA!\n")
cat("📁 Todos os resultados salvos na pasta 'output/'\n")
cat("📊 Parâmetros: delta =", round(mean(delta_samples), 2), 
    ", p =", round(mean(p_samples), 2), "\n")
cat("🎯 Acurácias:", round(acc_850$acuracia, 1), "%,", 
    round(acc_90$acuracia, 1), "%,", round(acc_8$acuracia, 1), "%\n")
