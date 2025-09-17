declare module "vite-plugin-svgr" {
  import { Plugin } from "vite";
  export default function svgrPlugin(...args: unknown[]): Plugin;
}
