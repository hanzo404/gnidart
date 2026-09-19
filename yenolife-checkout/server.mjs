import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));
const port = Number(process.env.PORT || 3000);
const publicFiles = new Set(['index.html', 'preview.css', 'woocommerce-checkout.css', 'woocommerce-cart.css', 'app.mjs', 'checkout.mjs']);
const types = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.svg': 'image/svg+xml', '.webp': 'image/webp', '.woff2': 'font/woff2', '.txt': 'text/plain; charset=utf-8',
};

const server = http.createServer(async (request, response) => {
  response.setHeader('X-Content-Type-Options', 'nosniff');
  response.setHeader('Referrer-Policy', 'no-referrer');
  response.setHeader('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; font-src 'self'; base-uri 'self'; form-action 'none'");
  // No host allowlist or frame restrictions: supports Arena's preview proxy.
  if (!['GET', 'HEAD'].includes(request.method)) {
    response.writeHead(405, { Allow: 'GET, HEAD', 'Content-Type': 'text/plain; charset=utf-8' });
    response.end('This design preview does not accept form data.');
    return;
  }
  try {
    const requestUrl = new URL(request.url, 'http://preview.invalid');
    const pathname = decodeURIComponent(requestUrl.pathname);
    const resource = pathname === '/' ? 'index.html' : pathname.slice(1);
    const isDownload = resource === 'woocommerce-checkout.css' && requestUrl.searchParams.get('download') === '1';
    const isAsset = /^assets\/[a-zA-Z0-9_-]+\.(woff2|webp|svg|txt)$/.test(resource);
    if (!publicFiles.has(resource) && !isAsset) {
      response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      response.end('Not found');
      return;
    }
    const file = resolve(root, resource);
    if (!file.startsWith(root.endsWith(sep) ? root : root + sep) || !(await stat(file)).isFile()) throw new Error('Not a public file');
    const content = await readFile(file);
    response.writeHead(200, {
      'Content-Type': types[extname(file)] || 'application/octet-stream',
      'Content-Length': content.length,
      'Cache-Control': isDownload ? 'no-store' : 'no-cache',
      ...(isDownload ? { 'Content-Disposition': 'attachment; filename="woocommerce-checkout.css"' } : {}),
    });
    response.end(request.method === 'HEAD' ? undefined : content);
  } catch {
    response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    response.end('Not found');
  }
});
server.listen(port, '0.0.0.0', () => console.log(`Yenolife checkout preview listening on 0.0.0.0:${port}`));
process.on('SIGTERM', () => server.close());
process.on('SIGINT', () => server.close());
