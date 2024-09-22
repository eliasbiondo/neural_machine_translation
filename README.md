# Neural Machine Translation with Attention

Este projeto implementa um modelo de tradução automática neural (NMT - Neural Machine Translation) com mecanismo de atenção (Attention) usando TensorFlow. O modelo é treinado para traduzir frases do português para o inglês, utilizando o dataset de TED Talks.

## Setup

Primeiro, faça a importação das bibliotecas necessárias, como TensorFlow, TensorFlow Datasets, NumPy e Matplotlib.

## Dataset

O dataset utilizado é o `ted_hrlr_translate/pt_to_en`, que contém pares de frases em português e inglês extraídos de TED Talks.

## Tokenização

A tokenização converte as frases em uma sequência de tokens, que são processadas pelo modelo. Utilizamos um tokenizador pré-treinado para o português e inglês.

## Modelo

O modelo segue a arquitetura Transformer, com múltiplas camadas de atenção e feed-forward. Ele é composto por:

- Encoder: responsável por processar a sequência de entrada (português).
- Decoder: gera a sequência de saída (inglês), utilizando a sequência processada pelo encoder e os tokens gerados anteriormente.
- Mecanismo de atenção: permite que o modelo foque em diferentes partes da sequência de entrada ao gerar cada token de saída.

## Hiperparâmetros

Alguns dos principais hiperparâmetros do modelo são:

- `num_layers`: número de camadas no encoder e decoder.
- `d_model`: dimensão dos embeddings.
- `num_heads`: número de cabeças no mecanismo de atenção.
- `dff`: número de unidades na camada feed-forward.
- `dropout_rate`: taxa de dropout para regularização.

## Treinamento

O modelo é treinado por 20 épocas, e checkpoints são salvos a cada 5 épocas. O otimizador utilizado é o Adam, com um agendador de taxa de aprendizado personalizado.

### Percepções pessoais

Durante o treinamento, observei que, na minha máquina local, o tempo necessário para processar cada batch era extremamente elevado, em torno de 10 minutos por etapa. Isso tornou o treinamento inviável localmente, especialmente devido ao tamanho do dataset e à complexidade do modelo Transformer. Para contornar essa limitação, optei por utilizar o Google Colab com uma GPU A100, o que acelerou significativamente o processo. Com a A100, o tempo de treinamento por batch foi reduzido drasticamente, permitindo que o modelo fosse treinado de maneira muito mais eficiente. No entanto, mesmo utilizando o Colab, é necessário estar atento ao tempo limite de uso da GPU e às desconexões ocasionais, que podem interromper o treinamento. Por isso, tornou-se essencial salvar checkpoints com frequência para garantir que o progresso não fosse perdido em caso de interrupções.
#### Miscelânea

- **Tokenização**: A etapa de tokenização é fundamental para garantir que o modelo entenda a estrutura das frases. A tokenização pré-treinada fornecida pelo TensorFlow simplifica bastante este processo, economizando tempo e evitando a necessidade de criar um tokenizador do zero. No entanto, um ponto negativo é que, se o vocabulário do dataset for muito diferente do vocabulário do tokenizador pré-treinado, pode haver uma perda de precisão, especialmente em expressões idiomáticas ou termos menos comuns.

- **Atenção**: O mecanismo de atenção é o que diferencia os Transformers dos modelos anteriores de tradução automática. Ele permite que o modelo foque nas partes mais relevantes da frase de entrada ao gerar a tradução, o que melhora a qualidade das traduções, especialmente em frases complexas. No entanto, o custo computacional do mecanismo de atenção é elevado, especialmente quando se trabalha com frases longas, o que aumenta o tempo de treinamento e a necessidade de hardware mais potente.

- **Desempenho**: O uso de GPUs poderosas, como a A100 no Colab, é essencial para treinar modelos complexos como este em um tempo razoável. Treinar em CPUs ou GPUs menos potentes pode ser inviável devido ao tempo necessário e ao risco de esgotamento de memória. A necessidade de hardware especializado pode ser uma barreira para desenvolvedores que não têm acesso a esses recursos. Além disso, mesmo com uma GPU poderosa, o tempo de treinamento ainda pode ser significativo, dependendo do número de camadas e da dimensão dos embeddings.

- **Checkpointing**: O salvamento de checkpoints a cada 5 épocas permitiu que o treinamento fosse retomado facilmente em caso de interrupções, o que foi muito útil ao usar o Colab, onde a conexão pode ser instável. No entanto, a configuração de checkpoints pode consumir espaço de armazenamento e exige um gerenciamento cuidadoso para evitar que o disco seja preenchido rapidamente. Além disso, se o modelo for interrompido muito frequentemente, o tempo gasto para salvar e carregar checkpoints pode se tornar um gargalo.

- **Escalabilidade**: Uma vantagem clara do Transformer é sua escalabilidade. Ao aumentar o número de camadas, cabeças de atenção e a dimensão dos embeddings, o modelo pode aprender representações mais complexas e gerar traduções mais precisas. No entanto, isso também aumenta significativamente o custo computacional. Em experimentos com configurações maiores, como 6 ou mais camadas no encoder e decoder, o modelo apresentou melhorias na qualidade das traduções, mas o tempo de treinamento aumentou consideravelmente.

- **Qualidade das Traduções**: As traduções geradas pelo modelo após o treinamento foram bastante satisfatórias, especialmente em frases simples e diretas. No entanto, em frases mais longas ou com estruturas gramaticais complexas, o modelo às vezes falhou em capturar nuances ou gerou traduções menos precisas. Isso indica que, embora o Transformer seja poderoso, ainda há espaço para ajustes finos, como o uso de datasets maiores ou mais específicos para melhorar a generalização.

- **Facilidade de Implementação**: A implementação do Transformer no TensorFlow foi facilitada pelo uso de tutoriais e ferramentas pré-existentes. A integração com o TensorFlow Datasets e o suporte a tokenizadores prontos economizou muito tempo. No entanto, a curva de aprendizado para entender profundamente o funcionamento dos Transformers, especialmente o mecanismo de atenção, pode ser íngreme para iniciantes.

- **Limitações de Recursos**: Embora o treinamento no Colab tenha sido uma solução viável, o tempo de uso da GPU é limitado, e a desconexão automática após um período de inatividade pode ser frustrante. Além disso, o Colab Pro oferece mais tempo de GPU, mas ainda assim é necessário monitorar o uso de recursos para evitar interrupções inesperadas.

Em resumo, o projeto apresentou resultados promissores, mas também destacou a importância de se ter acesso a hardware adequado e a necessidade de um gerenciamento eficiente de recursos, como checkpoints e tempo de GPU.
## Inferência

Após o treinamento, o modelo pode ser utilizado para traduzir novas frases. A classe `Translator` foi implementada para facilitar o processo de inferência. Basta fornecer uma frase em português, e o modelo retorna a tradução em inglês.

## Exemplo de Tradução

Aqui está um exemplo de tradução gerada pelo modelo:

- **Entrada**: "Este é um problema que temos que resolver."
- **Predição**: "this is a problem we have to solve."
- **Verdadeiro**: "this is a problem we have to solve."

## Requisitos

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- TensorFlow Datasets

## Como Executar

1. Clone o repositório.
2. Instale os requisitos.
3. Execute o notebook `neural_machine_translation_with_attention.ipynb`.

```bash
git clone https://github.com/eliasbiondo/neural_machine_translation.git
cd neural_machine_translation
pip install -r requirements.txt
jupyter notebook neural_machine_translation_with_attention.ipynb
```

## Conclusão

Este projeto mostra como utilizar a arquitetura Transformer para construir um modelo de tradução automática neural. A implementação inclui tokenização, treinamento com checkpoints, e um pipeline de inferência para gerar traduções de novas frases.

## Referências

- Tutorial oficial do TensorFlow sobre Transformers: [https://www.tensorflow.org/text/tutorials/transformer?hl=en](https://www.tensorflow.org/text/tutorials/transformer?hl=en)
