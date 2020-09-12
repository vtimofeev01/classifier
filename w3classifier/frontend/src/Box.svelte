<script>
	export let cls = 'box'
</script>
<style>


	.box {
		float: left;
		display: block;
		border: 1px solid #aaa;
		border-radius: 2px;
		box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
		padding: 0.5em;
		/*padding: 1em;*/
		/*margin: 0 0 1em 0;*/
		margin: 3px;
		min-width: 550px;
	}

	.box_instance {

		/*width: 300px;*/
		float: left;
		display: block;
		border: 1px solid #aaa;
		border-radius: 2px;
		box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
		padding: 1em;
		/*margin: 0 0 1em 0;*/
		margin: 3px;
		min-width: 550px;
		/*background-color: #cba39e;*/
	}
	.box_yellow {

		/*width: 300px;*/
		float: left;
		display: block;
		border: 1px solid #aaa;
		border-radius: 2px;
		box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
		padding: 1em;
		/*margin: 0 0 1em 0;*/
		margin: 3px;
		min-width: 550px;
		background-color: #e0d5d3;
	}

</style>

<div class={cls}>
	<slot></slot>
</div>