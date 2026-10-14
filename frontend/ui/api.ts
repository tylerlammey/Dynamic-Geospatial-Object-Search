/** Session bootstrap and authenticated JSON transport for the loopback service. */
const fragment = location.hash.slice(1);
if (fragment) sessionStorage.setItem('session-token', fragment);
export const token = fragment || sessionStorage.getItem('session-token') || '';
history.replaceState(null, '', location.pathname + location.search);

export function authenticatedUrl(route: string): string {
  return `${route}${route.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}`;
}

export async function api<T>(route: string, value?: unknown): Promise<T> {
  const response = await fetch(route, {
    method: value === undefined ? 'GET' : 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      ...(value === undefined ? {} : {'Content-Type': 'application/json'}),
    },
    body: value === undefined ? undefined : JSON.stringify(value),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Local service returned ${response.status}`);
  return data as T;
}
