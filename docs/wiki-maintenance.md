# Maintaining the GitHub Wiki

The Markdown files in `docs/wiki/` are the reviewed source for Exacta's GitHub Wiki. They
include a home page, task guides, sidebar and footer. Develop them alongside features on
short-lived branches and merge into `main`. Publishing Wiki pages does not change the code
repository's `main`. Keep the Wiki's public version aligned with the latest PyPI release.

GitHub requires an initial page to be saved through its website before the Wiki can be cloned.
If the wiki repository is unavailable, enable Wikis in the repository's settings, open
`https://github.com/rodolfo-viana/exacta/wiki`, and create a page titled `Home`. See
[GitHub's documentation](https://docs.github.com/en/communities/documenting-your-project-with-wikis/adding-or-editing-wiki-pages).

Once initialized, review and publish from the code repository root using SSH and the owner's
Git identity. This copies the managed pages and preserves any additional pages already present:

```bash
wiki_checkout=$(mktemp -d /tmp/exacta-wiki.XXXXXX)
git clone git@github.com:rodolfo-viana/exacta.wiki.git "$wiki_checkout"
git -C "$wiki_checkout" config user.name 'Rodolfo Viana'
git -C "$wiki_checkout" config user.email 'eu@rodolfoviana.com.br'
cp docs/wiki/*.md "$wiki_checkout/"
git -C "$wiki_checkout" add -- '*.md'
git -C "$wiki_checkout" diff --cached --check
git -C "$wiki_checkout" diff --cached --stat
# Review the staged pages before publishing.
git -C "$wiki_checkout" commit -m 'Update Exacta documentation'
git -C "$wiki_checkout" push origin HEAD
```

Only the wiki repository's default branch is displayed by GitHub. Review its remote history
before overwriting a managed page edited through the website. If the staged diff is empty,
there is nothing to commit. After publishing, inspect the home page, sidebar and links.
At each release, update the documented version and remove development labels only for
features included in that release.
