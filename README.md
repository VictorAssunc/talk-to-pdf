# Talk to PDF

Assistente Conversacional capaz de interpretar PDFs desenvolvido para a disciplina de Tópicos III - PUC Minas

## Dependências

- Python >= 3.12

## Como rodar localmente?

Para rodar, são necessárias duas variáveis de ambiente, que não serão expostas aqui por motivos de privacidade. As chaves necessárias são:
- Pinecone: É necessário criar uma conta e uma chave de API (https://app.pinecone.io)
- Google: É necessário criar uma conta e uma chave de API (https://aistudio.google.com)

Após criadas, é necessário executar o comando abaixo na raiz do projeto, substituindo os placeholders (`xxxxxx` e `yyyyyy`) pelas chaves geradas:
```
cat > .env<< EOF
PINECONE_API_KEY=xxxxxx
GOOGLE_API_KEY=yyyyyy
EOF
```

### Com make

`make deps run`

### Sem make

```
pip install -r requirements.txt
streamlit run main.py
```
