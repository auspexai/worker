// Cloudflare Worker for getworker.auspexai.network
//
// Proxies the install script from GitHub instead of redirecting,
// so there's no CDN caching staleness. Deploy via Workers & Pages
// dashboard and route to getworker.auspexai.network/*.

const SCRIPT_URL =
  "https://raw.githubusercontent.com/auspexai/worker/main/packaging/install.sh";

export default {
  async fetch() {
    const upstream = await fetch(SCRIPT_URL, {
      cf: { cacheTtl: 60, cacheEverything: true },
    });

    return new Response(await upstream.text(), {
      status: upstream.status,
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "public, max-age=60",
      },
    });
  },
};
