# Transfer Learning com MobileNetV2 para Classificação de Imagens

Este projeto utiliza **Transfer Learning** com o modelo pré-treinado **MobileNetV2** para classificar imagens de gatos e cachorros usando o dataset **Cats vs Dogs** do TensorFlow.

## Estrutura do Código

### Importação de Bibliotecas
O projeto inicia com a importação das bibliotecas necessárias:
- `tensorflow`: Biblioteca principal para deep learning.
- `tensorflow.keras`: Para criar e treinar o modelo.
- `tensorflow_datasets`: Para carregar datasets prontos.
- `matplotlib`: Para visualizações e análise.
- `os`: Manipulação de diretórios.
- `numpy`: Operações em arrays.

### Carregamento e Preprocessamento do Dataset
1. **Carregar o Dataset:**
   ```python
   dataset, info = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)
   ```
   Isso carrega o dataset `cats_vs_dogs` junto com suas informações.

2. **Dividir o Dataset:**
   - Usamos os primeiros 20.000 exemplos para treino e o restante para validação:
     ```python
     train_data = dataset['train'].take(20000)
     val_data = dataset['train'].skip(20000)
     ```

3. **Preprocessar Imagens:**
   - Redimensionamos para o tamanho 224x224 e normalizamos os pixels:
     ```python
     def preprocess(image, label):
         image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
         return image, label
     ```
   - Aplicamos o preprocessamento e organizamos os dados em lotes:
     ```python
     train_data = train_data.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
     val_data = val_data.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
     ```

### Construção do Modelo
1. **Modelo Base:**
   - Usamos o **MobileNetV2**, pré-treinado no ImageNet:
     ```python
     base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
     base_model.trainable = False
     ```

2. **Camadas Personalizadas:**
   - Adicionamos camadas para reduzir as dimensões e realizar classificação binária:
     ```python
     model = Sequential([
         base_model,
         GlobalAveragePooling2D(),
         Dropout(0.3),
         Dense(1, activation='sigmoid')
     ])
     ```

3. **Compilar o Modelo:**
   - Configuramos o otimizador, a função de perda e as métricas:
     ```python
     model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
     ```

### Treinamento do Modelo
Treinamos o modelo por 10 épocas, com dados de validação:
```python
history = model.fit(train_data, validation_data=val_data, epochs=10)
```

### Avaliação do Modelo
Após o treinamento, avaliamos o desempenho nos dados de validação:
```python
val_loss, val_acc = model.evaluate(val_data)
print(f'Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
```

### Visualização das Curvas de Aprendizado
As curvas de acurácia de treino e validação são plotadas para analisar o desempenho:
```python
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Epocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
```

### Teste com Novas Imagens
1. **Visualizar Previsões em Lote:**
   - Pega um lote do conjunto de validação e faz previsões:
     ```python
     for images, labels in val_data.take(1):
         predictions = model.predict(images)
         break
     ```
   - Mostra as imagens com as classes reais e previstas:
     ```python
     for i in range(len(images)):
         plt.imshow(images[i].numpy())
         plt.axis('off')
         plt.title(f"Real: {'Cachorro' if labels[i].numpy() == 1 else 'Gato'} | Previsto: {'Cachorro' if predictions[i] > 0.5 else 'Gato'}")
         plt.show()
     ```

2. **Prever em uma Nova Imagem:**
   - Carrega e processa a imagem:
     ```python
     img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
     img_array = image.img_to_array(img) / 255.0
     img_array = np.expand_dims(img_array, axis=0)
     ```
   - Faz a previsão e exibe o resultado:
     ```python
     prediction = model.predict(img_array)
     plt.imshow(img)
     plt.axis('off')
     plt.title(f"Previsto: {'Cachorro' if prediction[0] > 0.5 else 'Gato'}")
     plt.show()
     ```

## Requisitos
- **TensorFlow 2.x**
- **TensorFlow Datasets**
- **Matplotlib**
- **NumPy**

## Resultados
O modelo alcança alta acurácia nos dados de validação e demonstra boa capacidade de generalização ao classificar novas imagens. As visualizações mostram a correspondência entre as classes reais e previstas.

## Observação
Este projeto pode ser adaptado para outros datasets ou classes diferentes, ajustando o dataset de entrada e a camada de saída do modelo.

