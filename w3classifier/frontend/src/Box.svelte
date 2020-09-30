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
		margin: 3px;
		width: 95%;
	}

	.box_instance {

		float: left;
		display: block;
		border: 1px solid #aaa;
		border-radius: 2px;
		box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
		padding: 1em;
		/*margin: 0 0 1em 0;*/
		margin: 3px;
		width: 90%;
	}
	.box_yellow {

		float: left;
		display: block;
		border: 1px solid #aaa;
		border-radius: 2px;
		box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
		padding: 1em;
		margin: 3px;
		width: 90%;
		background-color: #e0d5d3;
	}

	.box_gray {
		float: left;
		display: block;
		border: 1px solid #aaa;
		border-radius: 2px;
		box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
		padding: 1em;
		margin: 3px;
		background-color: #acacac;
	}

	.thin_box {
		float: left;
		display: block;
		border: 1px solid #aaa;
		border-radius: 2px;
		box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
		padding: 4px;
		margin: 1px;
		background-color: #f1f1f1;
	}

</style>

<div class={cls}>
	<slot></slot>
</div>